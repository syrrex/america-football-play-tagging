#!/bin/bash
#SBATCH --job-name=kp_qlc_%j
#SBATCH --partition=iml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=kp_qlc_%j.out
#SBATCH --error=kp_qlc_%j.err
#SBATCH --exclude=nvcluster-node3

# Submit one job per fold:
#   sbatch --export=ALL,FOLD=Q1 run_quarter_lc.sh
#   sbatch --export=ALL,FOLD=Q2 run_quarter_lc.sh
#   sbatch --export=ALL,FOLD=Q3 run_quarter_lc.sh
#   sbatch --export=ALL,FOLD=Q4 run_quarter_lc.sh

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Fold:      $FOLD"
echo "Started:   $(date)"
echo "------------------------------"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

export TB2026_SRC=~/keypoint_dataset/tb2026_all
export KP_META=~/kp_experiment/tb2026_dataset_metadata.csv

python ~/kp_experiment/quarter_lc_training.py \
    --fold $FOLD \
    --device 0 \
    --imgsz 1280

echo "------------------------------"
echo "Finished training: $(date)"
echo "------------------------------"

# Evaluate all 6 levels for this fold
BASE=~/kp_experiment/quarter_lc_results_1280
VAL_IMGS=$BASE/$FOLD/val_images
VAL_LBLS=$BASE/$FOLD/val_labels

for LEVEL in 0.5Q 1.0Q 1.5Q 2.0Q 2.5Q 3.0Q; do
    WEIGHTS=$BASE/$FOLD/$LEVEL/train/weights/best.pt
    if [ -f "$WEIGHTS" ]; then
        echo "Evaluating $FOLD / $LEVEL ..."
        python ~/kp_experiment/evaluate_keypoints.py \
            --weights $WEIGHTS \
            --out     $BASE/$FOLD/$LEVEL/kpt_eval.json \
            --img-dir $VAL_IMGS \
            --lbl-dir $VAL_LBLS
    else
        echo "  [skip] $FOLD/$LEVEL — no weights found"
    fi
done

echo "------------------------------"
echo "Cleaning up val symlinks..."
rm -rf $BASE/$FOLD/val_images $BASE/$FOLD/val_labels
echo "All done: $(date)"