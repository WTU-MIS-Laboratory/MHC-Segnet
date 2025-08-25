# [MHC-Segnet](https://github.com/WTU-MIS-Laboratory/MHC-Segnet)

Official repository for "MHC-Segnet: Mamba–Hadamard collaboration segmentation network for multimodal MRI brain tumor".

[[Paper]](https://link.springer.com/article/10.1007/s00371-025-03961-2)


## Release

-  🔥**News**: ```2025/8/25```: MHC-Segnet released.


## Get Start

Requirements: `CUDA ≥ 11.8`
Minimize VRAM: 24G

1. Create a virtual environment: `conda create -n MHCSegnet python=3.8 -y` and `conda activate MHCSegnet`
2. Install Pytorch ≥ 2.2, torchvision ≥ 0.17.0, torchaudio ≥ 2.2.0
3. MONAI == 1.3.0, mamba_ssm == 1.2.0, causal_conv1d == 1.2.0. [[Mamba]](https://github.com/state-spaces/mamba)
4. Download code: `git clone https://github.com/WTU-MIS-Laboratory/MHC-Segnet.git`


## Data Preprocess

Download BraTS2019 dataset [here](https://www.med.upenn.edu/cbica/brats-2019/). Then unzip them into `datasets/processed`, and change the `'root'` path into your unzip path, 

```
train_set = {
        'root': 'MICCAI_BraTS_2019_Data_Training/Train',
        'file_list': 'train.txt',
        }
```


make sure the file tree of datasets as follow:


```
datasets/
├── processed/
│   ├── MICCAI_BraTS_2019_Data_Training/
│   │   ├── LGG
│   │   │   ├── BraTS19_TMC_30014_1
│   │   │   │	├── BraTS19_TMC_30014_1_t1.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_t1cd.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_t2.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_flair.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_seg.nii.gz
│   │   │   │	├── BraTS19_TMC_30014_1_pkl_ui8f32b0.pkl
│   │   │   ├── ...
│   │   ├── HGG
│   │   │   ├── ...
│   │   ├── cross_validation
│   │	│	├── t1.txt
│   │	│	├── t2.txt
│   │	│	├── ...
│   │	│	├── v1.txt
│   │	│	├── v2.txt
│   │	│	├── ...
│   ├── ...
```

## Train cli example
```
torchrun --nproc_per_node=1 --master_port=27681 Jtrain.py --gpus 4 --num_workers 16 --criterion BCEWithDiceLoss --batch_size 2 --lr 0.001 --optimizer AdamW --weight_decay 1e-2 --end_epoch 298 --task BraTS --datasets_dir 'Replace with the actual absolute path of the dataset' --experiment exp_name --train_file train.txt --valid_on_train 0 --valid_per_epoch 1 --valid_file valid.txt
```

