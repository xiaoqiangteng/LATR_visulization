#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29504 main.py \
#     --config config/release_iccv/apollo_standard.py \
#     --cfg-options evaluate=true \
#     eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/apollo/release_iccv/apollo_standard/model_best_epoch_235.pth

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29504 main.py \
#     --config config/release_iccv/apollo_rare.py \
#     --cfg-options evaluate=true \
#     eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/apollo/release_iccv/apollo_rare/model_best_epoch_145.pth

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29504 main.py \
    --config config/release_iccv/apollo_illu.py \
    --cfg-options evaluate=true \
    eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/apollo/release_iccv/apollo_illu/model_best_epoch_102.pth
