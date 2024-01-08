#!/usr/bin/env bash
#SBATCH -N 1
##SBATCH --array=0-2
#SBATCH -J colmap
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --constraint=[v100]
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=320G
##SBATCH  --cpus-per-task=64
##SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
##SBATCH --mail-user=jinjie.mai@kaust.edu.sa
##SBATCH  --account conf-2022-cvpr-ghanembs

# The project folder must contain a folder "images" with all the images.

EGO4D_ROOT=/ibex/ai/reference/CV/Ego4D/ego4d_data/v1/full_scale

# * note: you should get vq_val.json and vq_test_unannotated.json from Ego4D episodic memory official repo,
# as we point out in readme

python convert_videos_to_clips.py \
        --annot-paths data/vq_val.json data/vq_test_unannotated.json \
        --save-root data/clips \
        --ego4d-videos-root $EGO4D_ROOT \
        --num-workers 32