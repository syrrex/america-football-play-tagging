#!/bin/bash
#SBATCH --job-name=kp_upper_bound
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=kp_upper_bound_%j.out
#SBATCH --error=kp_upper_bound_%j.err

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

export KP_DS=~/keypoint_dataset

python ~/kp_experiment/upper_bound_experiment.py --device 0

echo "------------------------------"
echo "Finished:  $(date)"
