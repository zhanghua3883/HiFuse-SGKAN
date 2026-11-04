# HiFuse-SGKAN

Official PyTorch implementation of **HiFuse-SGKAN: Hierarchical Fusion Sparse-Grid Kolmogorov–Arnold Network for Medical Image Segmentation**.

## Environment



```
conda create -n hifuse_sgkan python=3.9 -y
conda activate hifuse_sgkan
pip install torch torchvision
pip install albumentations opencv-python scikit-learn pandas pyyaml tqdm
pip install tensorboardX thop medpy scipy
```

## Data Preparation

Arrange your dataset as follows:



```
inputs/<DatasetName>/
├── images/
│   ├── case_001.png
│   └── ...
└── masks/
    └── 0/
        ├── case_001.png
        └── ...
```

Supported datasets (auto-detected by name):



| `--dataset` | Image ext | Mask ext    |
| ----------- | --------- | ----------- |
| `cvc`       | `.png`    | `.png`      |
| `Kvasir`    | `.jpg`    | `.jpg`      |
| `busi`      | `.png`    | `_mask.png` |
| `glas`      | `.png`    | `.png`      |

## Training



```
python train.py \
    --arch UKAN \
    --dataset cvc \
    --data_dir inputs \
    --output_dir outputs \
    --name hifuse_sgkan_cvc \
    --num_classes 1 \
    --batch_size 8 \
    --epochs 400 \
    --lr 1e-4 \
    --kan_lr 1e-2 \
    --optimizer SGD \
    --loss BCEDiceLoss
```

The best checkpoint and training log (TensorBoard + CSV) are saved to `outputs/<name>/`.

## Evaluation



```
python val.py --output_dir outputs --name hifuse_sgkan_cvc
```

Prediction masks are saved under `outputs/<name>/out_val/`.

## Acknowledgement

This codebase is built upon [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet), [UKAN](https://github.com/BlurryLu/UKAN), and [UNet++](https://github.com/MrGiovanni/UNetPlusPlus). We sincerely thank the authors for their excellent work.

## Citation



```
@article{your2026hifusesgkan,
  title   = {HiFuse-SGKAN: Hierarchical Fusion Sparse-Grid Kolmogorov–Arnold Network for Medical Image Segmentation},
  author  = {Your Name},
  journal = {Journal Name},
  year    = {2026}
}
```