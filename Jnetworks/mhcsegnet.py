from __future__ import annotations
from collections.abc import Sequence

import time
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


from einops import rearrange, repeat
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None,None

try:
    from cprNetwork.mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

try:
    from cprNetwork.mamba.mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from cprNetwork.mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# by-ConvNeXt
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, input_x):
        if self.data_format == "channels_last":
            return F.layer_norm(input_x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = input_x.mean(1, keepdim=True)
            s = (input_x - u).pow(2).mean(1, keepdim=True)
            input_x = (input_x - u) / torch.sqrt(s + self.eps)
            input_x = self.weight[:, None, None] * input_x + self.bias[:, None, None]
            return input_x
# by-SegMamba
class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, input_x):
        input_x = self.fc1(input_x)
        input_x = self.act(input_x)
        input_x = self.fc2(input_x)
        return input_x
# by-Mamba
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v3",
        nslices=5
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.nslices = nslices

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        assert bimamba_type == "v3"
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        # assert bimamba_type == "v3"
        # spatial
        A_s = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_log = torch.log(A_s)  # Keep A_b_log in fp32
        self.A_s_log = nn.Parameter(A_s_log)
        self.A_s_log._no_weight_decay = True

        self.conv1d_s = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_s = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_s._no_weight_decay = True




        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v3":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                A_s = -torch.exp(self.A_s_log.float())

                xz_s = xz.chunk(self.nslices, dim=-1)
                xz_s = torch.stack(xz_s,dim=-1)
                xz_s = xz_s.flatten(-2)
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
                out_s = out_s.reshape(batch,self.d_inner,seqlen//self.nslices,self.nslices).permute(0,1,3,2).flatten(-2)

                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                out = F.linear(rearrange(out + out_b.flip([-1]) + out_s, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
# by-SegMamba
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out
# by-EGE-UNET
class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_in, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # B, C, H, W= x1.size()
        # ----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw----------#
        x4 = self.dw(x4)
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------norm----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x
# by-MedNeXt
class MedNeXtDownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,exp_r):
        super().__init__()

        self.dwconv1 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 2,
            padding = 3//2,
            groups = in_channels,
        )

        # Expansion
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.GELU()
        # Compression
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.res_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )

    def forward(self, x_residual,dummy_tensor=None):
        input_x = x_residual
        input_x = self.dwconv1(input_x)
        input_x = self.act(self.conv2(self.norm(input_x)))
        input_x = self.conv3(input_x)
        x_residual = self.res_conv(x_residual)
        return input_x + x_residual
# by-MedNeXt
class MedNeXtUpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,exp_r):
        super().__init__()

        self.dwconv1 = nn.ConvTranspose3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 2,
            padding = 3//2,
            groups = in_channels,
        )

        # Expansion
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.GELU()
        # Compression
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.res_conv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )

    def forward(self, x_residual,dummy_tensor=None):
        input_x = x_residual
        input_x = self.dwconv1(input_x)
        input_x = self.act(self.conv2(self.norm(input_x)))
        input_x = self.conv3(input_x)
        x_residual = self.res_conv(x_residual)
        input_x = input_x + x_residual
        input_x = torch.nn.functional.pad(input_x, (1, 0, 1, 0, 1, 0))
        return input_x


# by-self
class UnetrUpBlock_Seq_dummyFix(UnetrUpBlock):
    def forward(self, inp_skip, dummy_tensor=None, *args, **kwargs):
        inp, skip = inp_skip
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
# by-self
class UnetrUpBlock_nocat_dummyFix(UnetrUpBlock):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[int] | int,
            upsample_kernel_size: Sequence[int] | int,
            norm_name: tuple | str,
            res_block: bool = False,
    ):
        super(UnetrUpBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
    def forward(self, inp_nocat,dummy_tensor=None, *args, **kwargs):
        out = self.transp_conv(inp_nocat)
        out = self.conv_block(out)
        return out
# by-self
class UnetrBasicBlock_dummyFix(UnetrBasicBlock):
    def forward(self, inp, dummy_tensor=None, *args, **kwargs):
        return self.layer(inp)
# by-self
class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.conv_out = nn.ConvTranspose3d(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


# by-self
class Scale_aware_Feature_Refinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.nonliner = nn.GELU()
        self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
                )

        self.proj1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.proj3 = nn.Conv3d(in_channels, in_channels, 5, 1, 2, groups=in_channels)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x_residual, dummy_tensor=None):
        input_x = x_residual
        x1 = self.nonliner(self.norm(self.proj1(input_x)))
        x2 = self.nonliner(self.norm(self.proj2(input_x)))
        x3 = self.nonliner(self.conv(self.norm(self.proj3(input_x))))
        input_x = x1 + x2 + x3
        input_x =self.conv(input_x)
        return input_x + x_residual
# by-self
class Hierarchical_Fusion_Skip_Pathway(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.hfsp0 = UnetrBasicBlock_dummyFix(
            spatial_dims=3,
            in_channels=n_channels,
            out_channels=n_channels * 2,
            kernel_size=3,
            stride=2,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp1 = UnetrBasicBlock_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 2,
            out_channels=n_channels * 2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp2 = UnetrUpBlock_nocat_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 4,
            out_channels=n_channels * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp3 = UnetrUpBlock_nocat_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 8,
            out_channels=n_channels * 2,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name="instance",
            res_block=True,
        )
        self.fuconv = UnetrBasicBlock_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 2 * 4,
            out_channels=n_channels * 2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp_out0 = UnetrUpBlock_nocat_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 2,
            out_channels=n_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp_out1 = UnetrBasicBlock_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 2,
            out_channels=n_channels * 2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp_out2 = UnetrBasicBlock_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 2,
            out_channels=n_channels * 4,
            kernel_size=3,
            stride=2,
            norm_name="instance",
            res_block=True,
        )
        self.hfsp_out2_2 = UnetrBasicBlock_dummyFix(
            spatial_dims=3,
            in_channels=n_channels * 4,
            out_channels=n_channels * 8,
            kernel_size=3,
            stride=2,
            norm_name="instance",
            res_block=True,
        )

    def forward(self,input_x,dummy_tensor=None):
        input_x0, input_x1, input_x2, input_x3 = input_x
        input_x0 = self.hfsp0(input_x0)
        input_x1 = self.hfsp1(input_x1)
        input_x2 = self.hfsp2(input_x2)
        input_x3 = self.hfsp3(input_x3)
        input = torch.cat((input_x0,input_x1,input_x2,input_x3), dim=1)
        input = self.fuconv(input)
        output_x0 = self.hfsp_out0(input)
        output_x1 = self.hfsp_out1(input)
        output_x2 = self.hfsp_out2(input)
        output_x3 = self.hfsp_out2_2(output_x2)
        return output_x0, output_x1, output_x2, output_x3

# by-self
class SMEnc(nn.Module):
    # Scale_aware Mamba Encoder
    def __init__(self, in_channels,num_slices,exp_r) -> None:
        super().__init__()
        self.si = Scale_aware_Feature_Refinement(in_channels)
        self.mamba_layer = MambaLayer(dim=in_channels, num_slices=num_slices)
        # Expansion
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.GELU()
        # Compression
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self,x_residual:Tensor, dummy_tensor=None):
        fix = 0
        if fix == 1:
            input_x = x_residual
            input_x = self.si(input_x, dummy_tensor)
            input_x = self.mamba_layer(input_x)

            x_residual2 = input_x
            input_x = self.act(self.conv2(self.norm(input_x)))
            input_x = self.conv3(input_x)
        else:
            input_x = x_residual
            input_x = self.si(input_x, dummy_tensor)
            input_x = self.mamba_layer(input_x)

            x_residual2 = input_x
            input_x = self.act(self.conv2(self.norm(input_x)))
            input_x = self.conv3(input_x)
        return input_x + x_residual2
# by-self
class TGHPEnc(nn.Module):
    # Tri-Axis Hadamard Product Encoder
    def __init__(self,in_channels,exp_r):
        super().__init__()
        self.GHPA_dim = Grouped_multi_axis_Hadamard_Product_Attention(in_channels,in_channels)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Expansion
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.GELU()
        # Compression
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
    def forward(self, x_residual: Tensor, dummy_tensor=None):
        input_x = x_residual
        B,C,W,H,D = input_x.shape

        H_dim = self.GHPA_dim(input_x.permute(3 ,0, 1, 2, 4).reshape(-1,C,W,D))\
            .reshape(H,B,C,W,D).permute(1, 2, 3, 0, 4)
        W_dim = self.GHPA_dim(input_x.permute(2, 0, 1, 3 ,4).reshape(-1,C,H,D))\
            .reshape(W,B,C,H,D).permute(1, 2, 0, 3, 4)
        D_dim = self.GHPA_dim(input_x.permute(4, 0, 1, 2, 3).reshape(-1,C,W,H))\
            .reshape(D,B,C,W,H).permute(1, 2, 3, 4, 0)

        input_x = self.norm(W_dim + H_dim + D_dim)
        input_x = self.conv(input_x)

        input_x = input_x + x_residual
        x_residual2 = input_x
        input_x = self.act(self.conv2(self.norm(input_x)))
        input_x = self.conv3(input_x)
        return input_x + x_residual2

# by-self
def block_creator(encoder_str,depths_unidirectional,n_channels,depth_index,num_slices_default = 64):
    if encoder_str == "MambaEnc":
        block = nn.Sequential(*[
            SMEnc(in_channels=n_channels, num_slices=num_slices_default // (2 ** depth_index),
                  exp_r=depths_unidirectional)
            for _ in range(depths_unidirectional)
        ])
    elif encoder_str == "TGHPEnc":
        block = nn.Sequential(*[
            TGHPEnc(in_channels=n_channels,
                    exp_r=depths_unidirectional)
            for _ in range(depths_unidirectional)
        ])
    else:
        raise NotImplementedError
    return block


# by-self
class JCMNetv4(nn.Module):
    def __init__(self,
                 init_channels = 4,
                 n_channels = 32,
                 class_nums = 4,
                 depths_unidirectional = None,
                 checkpoint_style = "",
                 ):
        super(JCMNetv4, self).__init__()
        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "medium":
            depths_unidirectional = [1, 2, 3, 3, 4]
        elif depths_unidirectional == "large":
            depths_unidirectional = [2, 2, 4, 4, 4]
        elif depths_unidirectional == "small":
            depths_unidirectional = [1, 1, 2, 2, 2]

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False


        encoder = [ "MambaEnc", "MambaEnc","TGHPEnc", "TGHPEnc", "TGHPEnc"]

        # =============================

        self.stem = nn.Conv3d(init_channels, n_channels, kernel_size=1)

        self.repr_block_0 = block_creator(encoder[0],depths_unidirectional[0],n_channels,depth_index=0)

        self.dwn_block_0 = MedNeXtDownBlock(in_channels=n_channels,
                                            out_channels=n_channels * 2,
                                            exp_r=depths_unidirectional[0])


        self.repr_block_1 = block_creator(encoder[1],depths_unidirectional[1],n_channels * 2,depth_index=1)

        self.dwn_block_1 = MedNeXtDownBlock(in_channels=n_channels * 2,
                                            out_channels=n_channels * 4,
                                            exp_r=depths_unidirectional[1])


        self.repr_block_2 = block_creator(encoder[2],depths_unidirectional[2],n_channels * 4,depth_index=2)

        self.dwn_block_2 = MedNeXtDownBlock(in_channels=n_channels * 4,
                                            out_channels=n_channels * 8,
                                            exp_r=depths_unidirectional[2])


        self.repr_block_3 = block_creator(encoder[3],depths_unidirectional[3],n_channels * 8,depth_index=3)

        self.dwn_block_3 = MedNeXtDownBlock(in_channels=n_channels * 8,
                                            out_channels=n_channels * 16,
                                            exp_r=depths_unidirectional[3])

        self.emb_block = block_creator(encoder[4],depths_unidirectional[4],n_channels * 16,depth_index=4)

        self.hfsp_block = Hierarchical_Fusion_Skip_Pathway(n_channels)

        norm_name = ("group", {"num_groups": 1})
        self.repr_block_up_0 = UnetrUpBlock_Seq_dummyFix(
                spatial_dims=3,
                in_channels=n_channels * 2,
                out_channels=n_channels * 1,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.repr_block_up_1 = UnetrUpBlock_Seq_dummyFix(
                spatial_dims=3,
                in_channels=n_channels * 4,
                out_channels=n_channels * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.repr_block_up_2 = UnetrUpBlock_Seq_dummyFix(
                spatial_dims=3,
                in_channels=n_channels * 8,
                out_channels=n_channels * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.repr_block_up_3 = UnetrUpBlock_Seq_dummyFix(
                spatial_dims=3,
                in_channels=n_channels * 16,
                out_channels=n_channels * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.head = OutBlock(n_channels, class_nums)

        # Used to fix PyTorch checkpointing bug from MedNeXt
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    # by-MedNeXt
    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(x, l, self.dummy_tensor, use_reentrant=True)
        return x

    # by-self
    def forward(self, input: Tensor) -> Tensor:

        input = self.stem(input)

        if self.outside_block_checkpointing:
            repr0 = self.iterative_checkpoint(self.repr_block_0, input)
            dwn0 = checkpoint.checkpoint(self.dwn_block_0, repr0, self.dummy_tensor, use_reentrant=True)

            repr1 = self.iterative_checkpoint(self.repr_block_1, dwn0)
            dwn1 = checkpoint.checkpoint(self.dwn_block_1, repr1, self.dummy_tensor, use_reentrant=True)

            repr2 = self.iterative_checkpoint(self.repr_block_2, dwn1)
            dwn2 = checkpoint.checkpoint(self.dwn_block_2, repr2, self.dummy_tensor, use_reentrant=True)

            repr3 = self.iterative_checkpoint(self.repr_block_3, dwn2)
            dwn3 = checkpoint.checkpoint(self.dwn_block_3, repr3, self.dummy_tensor, use_reentrant=True)


            emb = self.iterative_checkpoint(self.emb_block, dwn3)


        else:

            repr0 = self.repr_block_0(input)
            dwn0 = self.dwn_block_0(repr0)

            repr1 = self.repr_block_1(dwn0)
            dwn1 = self.dwn_block_1(repr1)

            repr2 = self.repr_block_2(dwn1)
            dwn2 = self.dwn_block_2(repr2)

            repr3 = self.repr_block_3(dwn2)
            dwn3 = self.dwn_block_3(repr3)

            emb = self.emb_block(dwn3)

        skip0,skip1,skip2,skip3 = self.hfsp_block((repr0,repr1,repr2,repr3))

        repr3_dec = self.repr_block_up_3((emb,skip3))
        del skip3

        repr2_dec = self.repr_block_up_2((repr3_dec,skip2))
        del repr3_dec, skip2

        repr1_dec = self.repr_block_up_1((repr2_dec,skip1))
        del repr2_dec, skip1

        repr0_dec = self.repr_block_up_0((repr1_dec,skip0))
        del repr1_dec, skip0

        out = self.head(repr0_dec)
        del repr0_dec

        return out

class JCTest(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels = 64,
                 class_nums = 3,
                 depths_unidirectional="small",
                 checkpoint_style = "",
                 ):
        super(JCTest, self).__init__()

        self.test = UnetrUpBlock_nocat_dummyFix(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name="instance",
            res_block=True,
        )
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)


    def forward(self, input: Tensor) -> Tensor:
        out = checkpoint.checkpoint(self.test, input,self.dummy_tensor, input,use_reentrant=True)

        return out

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    cuda0 = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
    model = JCMNetv4(depths_unidirectional = 'small',checkpoint_style = 'outside_block')

    print(str(sum([param.nelement() for param in model.parameters()]) / 1e6) + 'M')
    time.sleep(3)

