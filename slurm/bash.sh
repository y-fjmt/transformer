#!/bin/bash

set -e

#SBATCH --gres=gpu:1
#SBATCH --nodelist=dgx-1
#SBATCH -i
#SBATCH --pty

srun --pty apptainer \
    shell \
    --nv \
    --bind .:/workspace,$HOME/.cache/huggingface:/root/.cache/huggingface \
    ./slurm/apptainer/pytorch2411.sif \
    /bin/bash -i