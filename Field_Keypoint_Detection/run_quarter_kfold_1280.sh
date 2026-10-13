#!/bin/bash
#SBATCH --job-name=kp_qkfold_1280
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=kp_qkfold_1280_%j.out
#SBATCH --error=kp_qkfold_1280_%j.err
#SBATCH --exclude=nvcluster-node3

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

export TB2026_SRC=~/keypoint_dataset/tb2026_all
export KP_META=~/kp_experiment/tb2026_dataset_metadata.csv

python ~/kp_experiment/quarter_kfold_training.py --device 0 --imgsz 1280

echo "------------------------------"
echo "Finished training: $(date)"
echo "------------------------------"

BASE=~/kp_experiment/quarter_kfold_results_1280

for Q in Q1 Q2 Q3 Q4; do
    echo "Evaluating $Q ..."
    python ~/kp_experiment/evaluate_keypoints.py \
        --weights $BASE/$Q/train/weights/best.pt \
        --out     $BASE/$Q/kpt_eval.json \
        --img-dir $BASE/$Q/val_images \
        --lbl-dir $BASE/$Q/val_labels
done

echo "------------------------------"
echo "Cleaning up val symlinks..."
for Q in Q1 Q2 Q3 Q4; do
    rm -rf $BASE/$Q/val_images $BASE/$Q/val_labels
done
echo "All done: $(date)"