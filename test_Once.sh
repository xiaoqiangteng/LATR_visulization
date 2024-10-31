#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29507 main.py \
    --config config/release_iccv/once.py \
    --cfg-options evaluate=true \
    eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/once/release_iccv/once/checkpoint_model_epoch_23.pth
