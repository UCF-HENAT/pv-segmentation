# PV-Segmentation
Pytorch Code for Generalized Deep Learning Model for Photovoltaic Module Segmentation from Satellite and Aerial Imagery

[Paper](https://pdf.sciencedirectassets.com/271459/1-s2.0-S0038092X24X00079/1-s2.0-S0038092X24002330/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGkaCXVzLWVhc3QtMSJIMEYCIQCaaIiJWugZVEFeoFk11ZzQHtJmeQ9lqKdbaWRRFBD5vQIhAOSGLpNSkzh4QwHzb0FFOWgFyzbZe7zC7E8161x4PBpyKrsFCJL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwvSTqMAWJky4iE%2FEQqjwW0fkZYiVnASvcln90cgkN8FRWMTjEt00JjYk%2Bwcu1w5HEV223TmGm90iQBv720uXwvWJcoLIK9vR63nqvrnFkeoZWkWBG28DqghffmdRNw2upqxTy8PFffTp%2BeBQdPXgqX1DDzVK2BjLUzs5lWRLyoiFdxwYK3bh8hmlf8PMjc8dqF1pzp%2FgfSn20aBRL8t7UGywtm4dm4TxdU%2BUB0AlkDqE6ZRDILuVZdRRREFcvc1NLWAyD5P7CtVLLUwE5Cid8GG%2FZZ5olqZsiUIh0JOOHaSpzdndJFRkhe9QVnlW6wTyPs%2BiLtKdMQaRawCYgZxkElfvqG9VxiuZK3e%2FVh79RDUqwg72Drfwu02CXDGjc5zW9HkFtzecDxXQ9HrjwkT7kaqhlVGkw1s9bZfCJBWaWJEWw3z7qFXP4EeWJyduFE94T62qI0%2BY0ccXbLnlVu0HTJ56khT%2Bpb2%2BC2lSTwGmnYvF%2FMdzBp4sjBPpJXlcrlEFeCXW8orDO%2BB4BkQGRWbVlWVAJ447fzVmxXUy0NB0MAIa4H3VBQGIVmOblCOFmAnvib%2F7TI1%2FOcsOPS9uKYHg1%2Bfxm88Mn7LW3MHlVGcimMTX2OoQdI4MQWYSqOk9ivkotmztGceaC0RQKTgl93yR%2Fbne%2BxesnG8%2BFqw1qBoTKV3Pe4R6QYjljAsfXVM0EiAGqHAyu2LWZa86SX8MGxY3lJkK%2B0X4yzUiMruIHXYsHrZd0xFe2Z2%2FUBao2oqE%2BFQvBJ7%2F7mMyPMRjWX4pdrjB2k1TazF93W6vWCighMPgWACKfsvgJtiPbby6MdsGDnm3aPsu9mSYKIH%2BlGjEP1BS0IV%2FnbzSkuURHvlt0fpNM0PVzTVwk8TulKI%2FkJ2o6uMKSSh7cGOrABgRK9UNmTwgzFRxloo3k%2BXitZMq01NIBbKU6ORtIMRCNoy2VZhMz5SCMeA4NmJc68l6yotwI4quqsmGjB06wfDVqzAkDQofdbKcqC7CcfIF4ADi1Tdmz9gD%2BIzlwjejG79hAmIRBahNSQHWvN2X7rE85h785IxD444ktGF7cPWWr37NlfRUe9B51pHrTL55FZtSCtVHLBPal1OBENeAXP5%2FngGlwW%2BeR3Z6GwJ3wTdSY%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240911T171309Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQUXRIVMR%2F20240911%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d03ffa8e8093496589876d17bad3425a557139934a649a317c8e90758139ff6e&hash=a8c8dae32a7d5dac6736d8e3de4448dbbfd38b858e57e0aadecd618adf2b4164&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0038092X24002330&tid=spdf-1a2ba83f-a6e6-4fe5-b006-8ced27f89297&sid=56259ca258723043cb4ad3b97a8a9a20d2f7gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f165f02575b575d56565b&rr=8c1949684bed31d8&cc=us)

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
