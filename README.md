# [SIMPLE OBB AUGMENTATION TOOL 简单的OBB框图像增广工具]
## Introduction
OBB(OrientedboundingBox) has been widly used in object detection. This tool provides a simple and light weight way to augment images that are labeled in OBB. The tool includes 6 augmenting methods:
* clip
* mixup
* rotate
* gauss noise
* gause blur
* denoise

And an visualize tool is attached in **visual.py** while the augmenting program is **augmentation.py**. Your image should be in order like `'1.jpg''2.jpg'`. And your pixel labels should be stored in a .txt file in this form `class_index x1 y1 x2 y2 x3 y3 x4 y4`

## Get Start
```
pip install -r requirements.txt
```
