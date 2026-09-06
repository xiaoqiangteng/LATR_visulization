# Environment

It is recommanded to build a new virtual environment.

## 1. Install pytorch and requirements.

```bash
# first install pytorch (CUDA 11.3 build; required for Ampere GPUs such as RTX 3090)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# then clone LATR and change directory to it to install requirements
cd ${LATR_PATH}
python -m pip install -r requirements.txt
```

> PyTorch 1.8.0 (CUDA 10.1) does **not** support Ampere GPUs (sm_86, e.g. RTX 3090).
> The code also relies on `batch_first` / `norm_first` (added in PyTorch 1.9), so
> `1.10.0+cu113` is the minimum compatible version.

## 2. Install mm packages

### 2.1 Install `mmcv`

```bash
# prebuilt wheel for torch 1.10.0 + CUDA 11.3
pip install mmcv-full==1.5.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

### 2.2 Install other mm packages

Install [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/openmmlab/mmdetection3d). Note that we use `mmdet==2.24.0` and `mmdet3d==1.0.0rc3`.
