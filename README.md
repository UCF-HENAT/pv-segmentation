# PV-Segmentation
Pytorch Code for Generalized Deep Learning Model for Photovoltaic Module Segmentation from Satellite and Aerial Imagery

## Installation

1. Clone this repository and navigate to pv-segmentation folder
```bash
git clone https://github.com/ucf-photovoltaics/pv-segmentation.git
cd pv-segmentation
```

2. Install Package
```Shell
conda create -n pv-segmentation python=3.10 -y
conda activate pv-segmentation
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Model weights

Download the model weights from the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main) repository and move them to the models folder.

- DeepLabV3+
- UNET
- Mask2Former

## Datasets
To run our experiments you need to download the following datasets:

-PV01, PV03, PV08: https://zenodo.org/records/5171712
-Google and IGN: https://zenodo.org/records/7358126

Once downloaded Google and IGN datasets, move them to the main datasets folder:

```
mv ~/pv-segmentation/datasets/bdappv/google ~/pv-segmentation/datasets/google
mv ~/pv-segmentation/datasets/bdappv/ign ~/pv-segmentation/datasets/ign
```

For PV01, PV03, and PV08 we need to convert the images from bmp to png, change the folder structure, and move them to the main datasets folder:

```
python preprocess.py
```

## Train and test model

python evaluate.py

## Citation
