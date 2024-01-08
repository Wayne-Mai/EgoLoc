#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -J colmap
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-113




# module load cuda
# conda init
# conda activate fd3d
# module load gcc

if [ $HOSTNAME == "<your local machine name>" ]; then

  # for debug
  SLURM_ARRAY_TASK_ID=0
  # test : 6c641082-044e-46a7-ad5f-85568119e09e
  python colmap_utils/get_camera_poses_colmap.py --taskid $SLURM_ARRAY_TASK_ID \
    --input_dir data/clips_frames \
    --output_dir data/colmap_modesl
else
  # if on the cluster
  python colmap_utils/get_camera_poses_colmap.py \
   --taskid $SLURM_ARRAY_TASK_ID 
fi
