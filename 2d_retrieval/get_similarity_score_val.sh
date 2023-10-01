#!/usr/bin/env bash



EXPT_ROOT=$PWD


CLIPS_ROOT=$VQ3D_ROOT/data/clips
VQ2D_ROOT=/path/to/your/VQ2D
VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data/val_annot_vq3d.json
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"
export HYDRA_FULL_ERROR=1
# export PATH="<PATH to conda environment binaries>:$PATH"

cd $VQ2D_ROOT


python similarity_score_val.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="val" \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=4 \
  data.debug_mode=True \
  logging.visualize=True \
  model.config_path="$VQ2D_ROOT/pretrained_models/slurm_8gpus_4nodes_baseline_v1_05_output/config.yaml" \
  model.checkpoint_path="$VQ2D_ROOT/pretrained_models/slurm_8gpus_4nodes_baseline_v1_05_output/model_final.pth" \
  logging.save_dir="$EXPT_ROOT/visual_queries_logs" \
  logging.stats_save_path="$EXPT_ROOT/visual_queries_logs/baseline_vq3d_detection_v105_demo.json" > vq3d_val.log

