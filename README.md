# PV-Segmentation
Pytorch Code for Generalized Deep Learning Model for Photovoltaic Module Segmentation from Satellite and Aerial Imagery

## Installation

1. Clone this repository and navigate to pv-segmentation folder (and clone mmsegmentation inside folder)
```bash
git clone https://github.com/ucf-photovoltaics/pv-segmentation.git
cd pv-segmentation
git clone https://github.com/open-mmlab/mmsegmentation.git
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

```
mkdir models
```

- DeepLabV3+
- UNET
- Mask2Former

## Datasets
To run our experiments you need to download the following datasets:

```
mkdir datasets
```

- [PV01, PV03, PV08](https://zenodo.org/records/5171712)
- [Google and IGN](https://zenodo.org/records/7358126)

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

```
mkdir work_dirs
python evaluate.py
```


## Citation
```
@article{GARCIA2024112539,
  title = {Generalized deep learning model for photovoltaic module segmentation from satellite and aerial imagery},
  journal = {Solar Energy},
  volume = {274},
  pages = {112539},
  year = {2024},
  issn = {0038-092X},
  doi = {https://doi.org/10.1016/j.solener.2024.112539},
  url = {https://www.sciencedirect.com/science/article/pii/S0038092X24002330},
  author = {Gustavo García and Alejandro Aparcedo and Gaurav Kumar Nayak and Tanvir Ahmed and Mubarak Shah and Mengjie Li},
  keywords = {Solar energy, PV panel detection, Segmentation, CNN, Mask2Former, Image processing},
  abstract = {As solar photovoltaic (PV) has emerged as a dominant player in the energy market, there has been an exponential surge in solar deployment and investment within this sector. With the rapid growth of solar energy adoption, accurate and efficient detection of PV panels has become crucial for effective solar energy mapping and planning. This paper presents the application of the Mask2Former model for segmenting PV panels from a diverse, multi-resolution dataset of satellite and aerial imagery. Our primary objective is to harness Mask2Former’s deep learning capabilities to achieve precise segmentation of PV panels in real-world scenarios. We fine-tune the pre-existing Mask2Former model on a carefully curated multi-resolution dataset and a crowdsourced dataset of satellite and aerial images, showcasing its superiority over other deep learning models like U-Net and DeepLabv3+. Most notably, Mask2Former establishes a new state-of-the-art in semantic segmentation by achieving over 95% IoU scores. Our research contributes significantly to the advancement solar energy mapping and sets a benchmark for future studies in this field.}
}
```
