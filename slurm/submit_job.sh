#!/bin/bash

set -e

HOME=/home/t121121

srun \
    --gres=gpu:8 \
    --nodelist=dgx-1 \
    apptainer \
    exec \
    --nv \
    --bind .:/workspace,$HOME/.cache/huggingface:/root/.cache/huggingface \
    ./slurm/apptainer/pytorch2411.sif \
    ./slurm/scripts/job.sh
