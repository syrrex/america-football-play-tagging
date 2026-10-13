#!/bin/bash
#SBATCH --job-name=kp_combined
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=kp_combined_%j.out
#SBATCH --error=kp_combined_%j.err
#SBATCH --exclude=nvcluster-node3

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

export KP_DATASETS=~/keypoint_dataset

python ~/kp_experiment/combined_training.py --device 0


echo "------------------------------"
echo "Finished:  $(date)"