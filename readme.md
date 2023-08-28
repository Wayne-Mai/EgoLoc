# EgoLoc: Revisiting 3D Localization from Egocentric Videos with Visual Queries (ICCV 2023) [Oral]
![image](https://github.com/Wayne-Mai/EgoLoc/assets/26301932/0158ab22-8ff6-48cc-9ed3-cb7bd5e5c0c1)

*The implementation of 1st winner for VQ3D challenge in Ego4D Workshop at CVPR 2023 and ECCV 2022.*

## Readme

1. This repo borrows some code from our previous work at [ECCVW 2022](https://github.com/Wayne-Mai/VQ3D_ECCVW2022).
2. Our validation report v1 is available on [arXiv](https://arxiv.org/abs/2212.06969).
3. TODO List
- [x] Init Code
- [ ] Add 2D Detector inference code
- [ ] Add MV-aggregation code
- [ ] Add pre-computed camera poses for validation and test set
- [ ] Add re-organized test code
- [ ] Set up Github pages


## Installation
Please follow [Ego4D VQ3D Benchmark](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D/README.md) to produce the required VQ2D result and baseline VQ3D results (e.g., video clip frames, camera intrinsics, camera poses from the baseline method) at first.

## RUN
After that, check the script `run_colmap.sh` we provide in `vq3d` to get COLMAP estimated poses.
## Registration
Use the following command to collect and register your COLMAP results to Matterport Scan coordinate system.

```
python register_colmap_baseline.py --merge_pose data/poses_json/all_clips_base_colmap.json --debug --align ra

```
The output `all_clips_base_colmap.json` will be your camera poses file.

## Evaluation and Test
Carefully follow the steps in [Ego4D VQ3D Benchmark](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D/README.md) to evaluate and test the results.
