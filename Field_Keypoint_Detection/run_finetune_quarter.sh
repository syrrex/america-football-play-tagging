#!/bin/bash
#SBATCH --job-name=kp_ft_%j
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=kp_ft_%j.out
#SBATCH --error=kp_ft_%j.err
#SBATCH --exclude=nvcluster-node3

# Submit one job per fold:
#   sbatch --export=ALL,FOLD=Q1 run_finetune_quarter.sh
#   sbatch --export=ALL,FOLD=Q2 run_finetune_quarter.sh
#   sbatch --export=ALL,FOLD=Q3 run_finetune_quarter.sh
#   sbatch --export=ALL,FOLD=Q4 run_finetune_quarter.sh

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Fold:      $FOLD"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

export TB2026_SRC=~/keypoint_dataset/tb2026_all
export KP_META=~/kp_experiment/tb2026_dataset_metadata.csv
export PRETRAIN_WEIGHTS=~/kp_experiment/cross_stadium_results/3/train/weights/best.pt

python ~/kp_experiment/finetune_quarter_training.py \
    --fold $FOLD \
    --device 0 \
    --imgsz 1280

echo "------------------------------"
echo "Finished training: $(date)"
echo "------------------------------"

BASE=~/kp_experiment/finetune_quarter_results_1280

echo "Evaluating $FOLD ..."
python ~/kp_experiment/evaluate_keypoints.py \
    --weights $BASE/$FOLD/train/weights/best.pt \
    --out     $BASE/$FOLD/kpt_eval.json \
    --img-dir $BASE/$FOLD/val_images \
    --lbl-dir $BASE/$FOLD/val_labels

echo "------------------------------"
echo "Cleaning up val symlinks..."
rm -rf $BASE/$FOLD/val_images $BASE/$FOLD/val_labels
echo "All done: $(date)"