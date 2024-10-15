#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 main.py --config config/release_iccv/latr_1000_baseline.py --cfg-options evaluate=true eval_ckpt=pretrained_models/openlane.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 main.py \
    --config config/release_iccv/latr_1000_baseline.py \
    --cfg-options evaluate=true \
    # eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/openlane/release_iccv/latr_1000_baseline/baseline/model_best_epoch_8.pth
    --eval_ckpt=/media/data3/txq/programmings/git/LATR_visulization/work_dirs/openlane/release_iccv/latr_1000_baseline/LATR_LNN_2/checkpoint_model_epoch_39.pth
