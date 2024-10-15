#!/bin/bash

CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node 3 --master_port 29505 main.py --config config/release_iccv/once.py
