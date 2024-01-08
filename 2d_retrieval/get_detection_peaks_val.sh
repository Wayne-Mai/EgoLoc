#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -J vq2d
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=80G

##SBATCH  --cpus-per-task=24
##SBATCH --array=0-2
##SBATCH --constraint=[v100]
##SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
##SBATCH --mail-user=jinjie.mai@kaust.edu.sa


# conda init bash
# conda activate ego4d_vq2d

# module load gcc/8.2.0
# module load cuda/10.2.89



EXPT_ROOT=$PWD

# module load anaconda3/2020.11
# module load cuda/10.2
# module load cudnn/v7.6.5.32-cuda.10.2
# module load gcc/7.3.0
# module load cmake/3.15.3/gcc.7.3.0

# model.config_path="$VQ2D_ROOT/pretrained_models/slurm_8gpus_4nodes_baseline_v1_05_output/config.yaml" \
# model.checkpoint_path="$VQ2D_ROOT/pretrained_models/slurm_8gpus_4nodes_baseline_v1_05_output/model_final.pth" \
# model.config_path="$VQ2D_ROOT/pretrained_models/siam_rcnn_residual/config.yaml" \
# model.checkpoint_path="$VQ2D_ROOT/pretrained_models/siam_rcnn_residual/model.pth" \


CLIPS_ROOT=VQ2D/data/clips
VQ2D_ROOT=VQ2D
VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data/val_annot_vq3d.json
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"
export HYDRA_FULL_ERROR=1
# export PATH="<PATH to conda environment binaries>:$PATH"

cd $VQ2D_ROOT

# You should get the pretrain model from Ego4D's official repo or website page, v1.05 might not be the latest one,
# you can use a newer model for better performance
python get_detection_peaks_val.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="val" \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=4 \
  data.debug_mode=True \
  logging.visualize=True \
  model.config_path="$VQ2D_ROOT/pretrained_models/slurm_8gpus_4nodes_baseline_v1_05_output/config.yaml" \
  model.checkpoint_path="$VQ2D_ROOT/pretrained_models/slurm_8gpus_4nodes_baseline_v1_05_output/model_final.pth" \
  logging.save_dir="$EXPT_ROOT/visual_queries_logs" \
  logging.stats_save_path="$EXPT_ROOT/visual_queries_logs/baseline_vq3d_detection_v105_val.json" > vq3d_val.log

