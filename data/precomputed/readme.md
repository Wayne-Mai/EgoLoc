This folder contains precomputed data from us, all of them are compressed in JSON format.
- *scan_to_intrinsics_ego4d.json* : Camera intrinsics computed by Ego4D VQ3D.IF you are using camera poses extracted by Ego4D VQ3D, please use this intrinsics.
- *scan_to_intrinsics_egoloc.json*: Camera intrinsics computed by ours, best performance is achieved by using intrinsics and camera poses together from ours.
- *vq2d_val_vq3d_peak_only.json*: Since VQ3D video clips are a subset of VQ2D video clips, we provide 2D detection results for VQ3D's val set in this file.