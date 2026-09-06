#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 main.py --config config/release_iccv/latr_1000_baseline.py --cfg-options evaluate=true eval_ckpt=pretrained_models/openlane.pth
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node 5 --master_port 29501 main.py \
    --config config/release_iccv/latr_1000_baseline.py \
    --cfg-options evaluate=true \
    eval_ckpt=/public/home/tengxiaoqiang/programmings/git/LATR_visulization/work_dirs/openlane/release_iccv/SalienceLane_diffusion_MDSA/model_best_epoch_10.pth
