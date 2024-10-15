#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29501 main.py --config config/release_iccv/apollo_standard.py
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29502 main.py --config config/release_iccv/apollo_rare.py
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29503 main.py --config config/release_iccv/apollo_illu.py
