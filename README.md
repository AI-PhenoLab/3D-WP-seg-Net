# Welcome use 3D WP-seg Net

Welcome to 3D Wheat Plot Detection, we will tell you how to use each module of the application, as well as the environment configuration of the software.

## Installation
To run.py files, you need to install the following libraries
```javascript
conda create -n wheatPlot python=3.8
conda activate wheatPlot
pip install -r requirements.txt
pip install -e .
```
## How to train a model
If you wish to train a model, the following steps can be taken.
```javascript
cd /File path/3D WP-seg Net
python tools/train.py {File Path}\3D WP-seg Net\pointnet2_ssg_2xb16-cosine-200e_scannet-seg-xyz-only.py
```

## Peroration
We mainly added CSAM.py to mmdet3d\models\backbones, and modified pointnet2_sa_msg_ori.py.The above represents merely the first version of the software, and suggestions will be gathered and sorted out in the future for the improvement of the software.


# AI-PheneLab

