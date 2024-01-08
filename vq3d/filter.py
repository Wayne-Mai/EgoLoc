import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R


def frame_filter(args, frames):
    if 'clip' or 'camera' not in args.experiment:
        return frames

    res = []
    clip_scores = [frame['clip_score'] for frame in frames]
    detection_scores = [frame['detection_score'] for frame in frames]

    clip_topk = np.argsort(clip_scores)  # min->max
    detection_topk = np.argsort(detection_scores)  # min->max
    if 'top1clip' in args.experiment:
        return [frames[clip_topk[-1]]]
    elif 'top5clip' in args.experiment:
        selected = sorted(clip_topk[-5:])
        for idx in selected:
            res.append(frames[idx])
        return res
    elif 'camera' in args.experiment:
        # do nothing
        return frames


def find_neighbors_cameras(all_cameras,
                           q_camera,
                           area_threshold=2,
                           view_threshold=0.5):
    all_cameras = torch.from_numpy(np.array(all_cameras))
    q_camera = torch.from_numpy(np.array(q_camera).reshape(-1, 4, 4))

    num_cams = all_cameras.shape[0]
    all_xyz = all_cameras[:, 0:3, 3]
    q_xyz = q_camera[:, 0:3, 3]
    # max_spread = all_xyz.max() - all_xyz.min()
    # location_inds = (torch.abs(all_xyz-q_xyz)).max(dim=1)[0] < max_spread*area_threshold
    location_inds = torch.norm(all_xyz - q_xyz, dim=1) < area_threshold
    all_r = all_cameras[:, 0:3, 0:3]
    q_r = q_camera[:, 0:3, 0:3]
    all_ang = torch.from_numpy(np.array(R.from_matrix(all_r).as_euler('zxy')))
    q_ang = torch.from_numpy(np.array(R.from_matrix(q_r).as_euler('zxy')))
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    direc_inds = cos(all_ang, q_ang.repeat(num_cams, 1))
    direc_inds = 0.5 * (direc_inds + 1.0) >= view_threshold
    return direc_inds * location_inds


def camera_filter(args, frame_indices_valid, poses):
    if 'camera' not in args.experiment:
        return frame_indices_valid

    last_peak = np.argmax(frame_indices_valid)

    all_cameras = [poses[i] for i in frame_indices_valid]
    q_camera = poses[frame_indices_valid[last_peak]]
    valid_cameras = find_neighbors_cameras(all_cameras, q_camera)

    invalid_count = 0
    res = []
    for idx, valid in enumerate(valid_cameras):
        if valid:
            res.append(frame_indices_valid[idx])
        else:
            invalid_count += 1
    print(
        f"Camera selection: we delete {invalid_count}/{len(frame_indices_valid)} views using {last_peak} "
    )
    assert len(res) != 0
    return res


def get_blurry_score(frame_path):
    img = cv2.imread(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clear_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return clear_score

