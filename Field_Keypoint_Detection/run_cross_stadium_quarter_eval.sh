#!/bin/bash
#SBATCH --job-name=kp_cs_eval
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=kp_cs_eval_%j.out
#SBATCH --error=kp_cs_eval_%j.err
#SBATCH --exclude=nvcluster-node3

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

export TB2026_SRC=~/keypoint_dataset/tb2026_all
export KP_META=~/kp_experiment/tb2026_dataset_metadata.csv

python ~/kp_experiment/cross_stadium_quarter_eval.py

echo "------------------------------"
echo "All done: $(date)"