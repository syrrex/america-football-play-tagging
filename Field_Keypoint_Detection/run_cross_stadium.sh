#!/bin/bash
#SBATCH --job-name=kp_cross_stadium
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=kp_cross_stadium_%j.out
#SBATCH --error=kp_cross_stadium_%j.err

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

# Point the script to the datasets on local node storage
export KP_DATASETS=~/keypoint_dataset

python ~/kp_experiment/cross_stadium_experiment.py --device 0

echo "------------------------------"
echo "Finished:  $(date)"
