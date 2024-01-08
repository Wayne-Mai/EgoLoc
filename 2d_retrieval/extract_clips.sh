#!/usr/bin/env bash
#SBATCH -N 1
##SBATCH --array=0-2
#SBATCH -J v2c
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=1:00:00
##SBATCH --gres=gpu:1
##SBATCH --constraint=[a100|v100]
##SBATCH --cpus-per-gpu=32
#SBATCH --mem=320G
#SBATCH  --cpus-per-task=32
##SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90


# The project folder must contain a folder "images" with all the images.

EGO4D_ROOT=/ibex/ai/reference/CV/Ego4D/ego4d_data/v1/full_scale

python convert_videos_to_clips.py \
        --annot-paths data/vq_val.json data/vq_test_unannotated.json \
        --save-root data/clips \
        --ego4d-videos-root $EGO4D_ROOT \
        --num-workers 32