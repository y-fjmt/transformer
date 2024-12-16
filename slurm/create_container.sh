#!/bin/bash

set -e

# create base apptainer image
apptainer build \
        --force \
        --fakeroot \
        slurm/apptainer/pytorch2411_base.sif \
        docker://nvcr.io/nvidia/pytorch:24.11-py3

# setup apptainer image
apptainer build \
        --force \
        --fakeroot \
        slurm/apptainer/pytorch2411_base.sif \
        slurm/apptainer/pytorch2411.def