#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -J Col3S
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-30




# module load cuda
# conda init
# conda activate fd3d
# module load gcc

if [ $HOSTNAME == "kw60112" ]; then

  # for debug
  SLURM_ARRAY_TASK_ID=0
  # test : 6c641082-044e-46a7-ad5f-85568119e09e
  python get_camera_poses_colmap.py --taskid $SLURM_ARRAY_TASK_ID \
    --input_dir data/v1/debug_clips_frames \
    --output_dir data/v1/local_wayne_colmap
else
  python get_camera_poses_colmap.py \
   --taskid $SLURM_ARRAY_TASK_ID 
fi
