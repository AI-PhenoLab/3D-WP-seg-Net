# Welcome use 3D WP-seg Net

Welcome to 3D WP-seg Net, we will tell you how to use each module of the application, as well as the environment configuration of the software.

## Installation
To run.py files, you need to install the following libraries
```javascript
conda create -n pointcept python=3.10 -y
conda activate pointcept

# (Optional) If no CUDA installed
conda install nvidia/label/cuda-12.4.1::cuda conda-forge::cudnn conda-forge::gcc=13.2 conda-forge::gxx=13.2 -y

conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu124

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
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

