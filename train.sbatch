#!/bin/bash
#SBATCH --job-name=ccil_dynamics
#SBATCH --output=logs/%x-%j/out.log
#SBATCH --error=logs/%x-%j/err.log
#SBATCH --partition=overcap
#SBATCH --gres=gpu:"a40":1
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --time=8:00:00


export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/atian31/flash8/miniconda3/etc/profile.d/conda.sh
cd /nethome/atian31/flash8/repos/CCIL/
conda activate ccil

python correct_il/train_dynamics_model.py config/libero_spatial.yml