#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 main.py --config config/release_iccv/latr_1000_baseline.py --cfg-options evaluate=true eval_ckpt=pretrained_models/openlane.pth
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 main.py \
    --config config/release_iccv/latr_1000_baseline.py \
    --cfg-options evaluate=true \
    eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/openlane/release_iccv/latr_1000_baseline/baseline/model_best_epoch_8.pth
