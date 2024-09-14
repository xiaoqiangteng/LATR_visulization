#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29501 main.py --config config/release_iccv/latr_1000_baseline.py
# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node 2 main.py --config config/release_iccv/latr_1000_baseline.py
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 main.py --config config/release_iccv/latr_1000_baseline.py
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 main.py --config config/release_iccv/latr_1000_baseline.py --load_from /path/to/checkpoint.pth
