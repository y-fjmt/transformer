#!/bin/bash
set -e

# python3 train.py

torchrun --nnodes=1 --nproc_per_node=8 train_ddp.py