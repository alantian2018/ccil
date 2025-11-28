#!/bin/bash
#SBATCH --job-name=libero_spatial
#SBATCH --gres=gpu:H200:1         
#SBATCH --mem-per-gpu=128G                # Memory per gpu
#SBATCH -t 8:00:00                     # Duration of the job
#SBATCH -o ./logs/%x-%j.out           # Combined output and error messages file, with job name


export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /home/hice1/atian31/scratch/miniconda3/etc/profile.d/conda.sh
cd /home/hice1/atian31/scratch/ccil
conda activate ccil

python correct_il/train_dynamics_model.py config/libero_spatial.yml
