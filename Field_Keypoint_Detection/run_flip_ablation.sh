#!/bin/bash
#SBATCH --job-name=kp_flip_ablation
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=kp_flip_ablation_%j.out
#SBATCH --error=kp_flip_ablation_%j.err

export KP_DS=~/keypoint_dataset

python ~/kp_experiment/flip_ablation.py --device 0
