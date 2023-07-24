from distutils.log import debug
import os
import sys
import json
import h5py
import torch
import argparse
import numpy as np
from PIL import Image
import copy
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import glob

import math

import numpy as np
import open3d as o3d
from sklearn import linear_model, datasets
# from spatialmath import *
import spatialmath.base as tr
import shortuuid
import random
# sys.path.append('../data')

# from scipy.spatial.transform import Rotation
"""
We should return:
{
    'image_id': filename_last_digit (int) [3x3] poses matrix
}
"""
debug_json = {}


def load_colmap_to_poses(filename=None):
    with open(filename) as f:
        lines = f.readlines()
    output = {}
    for index, temp_line in enumerate(lines):
        line = temp_line.strip()
        if not line.startswith('#') and (line.endswith('.png')
                                         or line.endswith('jpg')):
            temp = line.split(' ')

            qw = float(temp[1])
            qx = float(temp[2])
            qy = float(temp[3])
            qz = float(temp[4])

            tx = float(temp[5])
            ty = float(temp[6])
            tz = float(temp[7])

            name = temp[9]
            # * use it as key save it as dict
            pose_id = int(name.split('_')[-1].split('.')[0])

            # rotation_matrix = qvec2rotmat([qw, qx, qy, qz])
            # rotation_matrix = Rotations.from_quat([qw, qx, qy, qz]).as_matrix()
            rotation_matrix = tr.q2r([qw, qx, qy, qz])

            w2c = np.eye(4)
            w2c[:3, :3] = rotation_matrix
            w2c[:3, 3] = np.array([tx, ty, tz])

            output[pose_id] = w2c
    return output


"""
def do_align(pose_base, pose_colmap, align="median"):
    estimations = []
    for pb, pc in zip(pose_base, pose_colmap):
        estimations.append(np.linalg.solve(pc, pb))
        # print(estimations[-1])
    estimations = np.array(estimations)
    if align == "median":
        tcb = np.median(estimations, axis=0)
    else:
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(temp_poses_base, temp_poses_colmap)
        print(ransac.estimator_.coef_)

    error = np.linalg.norm(np.array(pose_colmap) @ tcb - np.array(pose_base),
                           axis=0)
    # error = np.max(np.array(pose_colmap) @ tcb - np.array(pose_base),axis=0)
    # error = np.array(pose_colmap) @ tcb - np.array(pose_base)
    return tcb, error
"""


def do_align(pose_base, pose_colmap, align, **kwargs):

    # rotation_base,rotation_colmap=[],[]
    # translation_base,translation_colmap=[],[]

    rcbs, tcbs, sep_rcbs, sep_tcbs = [], [], [], []
    for pb, pc in zip(pose_base, pose_colmap):
        solved_T = tr.trnorm(np.linalg.solve(pc, pb))
        rcbs.append(solved_T[:3, :3])
        tcbs.append(solved_T[:3, -1])
        if align == 'sep':
            sep_rcbs.append(tr.trnorm(np.linalg.solve(pc[:3, :3], pb[:3, :3])))
            sep_tcbs.append(pb[:3, -1] - pc[:3, -1])
            # colmap_translation + colmap_delta

    my_uuid = shortuuid.uuid()

    if align == 'median':
        optimal_R = tr.trnorm(np.median(rcbs, axis=0))
        optimal_t = np.median(tcbs, axis=0)
        error = 'unknown'
    elif align == 'random':
        idx = random.randrange(len(rcbs))
        optimal_R = tr.trnorm(rcbs[idx])
        optimal_t = tcbs[idx]
        error = 'unknown'
    elif align == 'sep':
        optimal_R, error, _ = rotation_average(sep_rcbs, **kwargs)
        optimal_t = np.median(sep_tcbs, axis=0)
    elif align == 'ra':
        optimal_R, error, _ = rotation_average(rcbs, **kwargs)
        optimal_t = np.median(tcbs, axis=0)
    else:
        raise NotImplementedError()

    debug_json[my_uuid] = {
        'optimal_R': optimal_R.tolist(),
        'rcbs': [x.tolist() for x in rcbs],
        'optimal_t': optimal_t.tolist(),
        'tcbs': [x.tolist() for x in tcbs]
    }

    res = np.empty([4, 4])
    res[:3, :3] = optimal_R
    res[:3, -1] = optimal_t
    res[3, :] = np.array([0, 0, 0, 1])

    return res, error, my_uuid


def rot_mul(R1, R2):
    return tr.trnorm(np.matmul(R1, R2))


def do_transform(R, optimal_R, align):
    if align == 'sep':
        res = np.empty([4, 4])
        res[:3, :3] = rot_mul(R[:3, :3], optimal_R[:3, :3])
        res[:3, -1] = R[:3, -1] + optimal_R[:3, -1]
        res[3, :] = np.array([0, 0, 0, 1])
        return res
    else:
        return np.matmul(R, optimal_R)


def rotation_average(rcbs, tolerance=1e-10, max_iter=100, vis=False):
    # rcbs=[SO3(x) for x in rcbs]

    def skew_matrix_2_vector(skew_matrix):
        return np.array(
            [skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

    def vector_2_skew_matrix(vector):
        return np.array([[0, -vector[2],
                          vector[1]], [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    optimal_R = rcbs[0]

    loop = 0
    while loop < max_iter:
        rvecs = [tr.trlog(rot_mul(optimal_R.T, rcb)) for rcb in rcbs]
        rvec_33 = np.mean(rvecs, axis=0)
        error = np.linalg.norm(skew_matrix_2_vector(rvec_33))
        if error < tolerance:
            break
        optimal_R = rot_mul(optimal_R, tr.trexp(rvec_33))
        loop += 1

    # test it on the matlab dataset
    if vis:
        tr.trplot(optimal_R, frame='A', width=1)
        for rcb in rcbs:
            # print(rcb)
            tr.trplot(tr.trnorm(rcb), frame='', color='green', width=0.5)

    return optimal_R, error, loop


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        default='data/clips_camera_poses/',
                        help="COLMAP camera pose folder")

    parser.add_argument(
        "--test_pose",
        type=str,
        default='data/poses_json/all_clips_camera_poses_test.json',
        help="Camera pose json folder")

    parser.add_argument(
        "--val_pose",
        type=str,
        default='data/poses_json/all_clips_camera_poses_val.json',
        help="Camera pose json folder")

    parser.add_argument("--new_pose",
                        type=str,
                        default='data/poses_json/all_clips_colmap.json',
                        help="Added colmap pose json")

    parser.add_argument("--miss_pose",
                        type=str,
                        default='data/poses_json/all_clips_pose_error.json',
                        help="Missed colmap pose json")

    parser.add_argument("--merge_pose",
                        type=str,
                        default='data/poses_json/all_clips_base_colmap.json',
                        help="Merged colmap pose json")

    parser.add_argument("--vq3d_clip_names",
                        type=str,
                        default='data/v1/3d/all_clips_for_vq3d_wayne.json',
                        help="Vq3d val and test clips names")

    parser.add_argument(
        "--align",
        type=str,
        default='sep',  # median, ra
        help='how to align colmap poses and baseline poses')

    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    with open(args.test_pose, 'r') as f:
        test_pose = json.load(f)
    with open(args.val_pose, 'r') as f:
        val_pose = json.load(f)

    with open(args.vq3d_clip_names, 'r') as f:
        all_clips_for_vq3d_wayne = json.load(f)

    args.merge_pose = args.merge_pose.split('.')
    args.new_pose=args.merge_pose[
        0] + '_' + args.align + '_newpose.' + args.merge_pose[1]
    args.merge_pose = args.merge_pose[
        0] + '_' + args.align + '.' + args.merge_pose[1]
    


    all_summary = {}
    all_res = {
        **json.load(open(args.val_pose, 'r')),
        **json.load(open(args.test_pose, 'r'))
    }
    # all_err={}
    for clip in tqdm(all_clips_for_vq3d_wayne['clips']):
        clip_uid = clip['clip_uid']
        all_summary[clip_uid] = {}
        # all_err[clip_uid]={}
        if clip['split'] == 'test':
            cur_base_pos = test_pose[clip_uid]
        else:
            cur_base_pos = val_pose[clip_uid]

        valid_frames = np.where(cur_base_pos['good_poses'])[0]
        all_summary[clip_uid]['stat_valid_base'] = valid_frames.tolist()
        # another key : camera_poses
        cur_base_pos['camera_poses'] = [
            np.concatenate((np.array(x), np.array([[0., 0., 0., 1.]])), axis=0)
            for x in cur_base_pos['camera_poses']
        ]

        colmap_paths = glob.glob(os.path.join(args.input_dir, clip_uid, '*/'))

        submap_log = "\n"
        valid_colmap_frames = 0
        for sidx, submap in enumerate(colmap_paths):
            # temp_output : {int(frame_id) : }
            temp_output = load_colmap_to_poses(
                os.path.join(submap, 'images.txt'))

            # note : camera poses are in w2c format
            valid_colmap_frames += len(temp_output.keys())

            temp_colmap_valid = []
            temp_colmap_invalid = []
            for frame_no in temp_output.keys():
                if frame_no in valid_frames:
                    temp_colmap_valid.append(frame_no)
                else:
                    temp_colmap_invalid.append(frame_no)
            submap_log += f"| map {sidx} : {len(temp_colmap_valid)}/{len(temp_output.keys())} | "

            # do the optimization for temp_colmap_valid
            if len(temp_colmap_valid) == 0: # common frame fno in this submap
                print(f"Failed at *{clip_uid}*{sidx}*{submap}")
            else:
                temp_poses_base = [
                    cur_base_pos['camera_poses'][x] for x in temp_colmap_valid
                ]
                temp_poses_colmap = [temp_output[x] for x in temp_colmap_valid]

                T_estimation, error, my_uuid = do_align(temp_poses_base,
                                                        temp_poses_colmap,
                                                        align=args.align,
                                                        tolerance=1e-10,
                                                        max_iter=100,
                                                        vis=False)  # tc -> tb
                # do something to add data here as well
                if args.debug:
                    debug_json[my_uuid]['clip_uid'] = clip_uid
                    debug_json[my_uuid]['map_id'] = submap
                    debug_json[my_uuid]['all_colmap_poses'] = [{
                        x:
                        temp_output[x].tolist()
                    } for x in temp_output]
                    debug_json[my_uuid]['base_poses'] = [
                        x.tolist() for x in temp_poses_base
                    ]
                    debug_json[my_uuid]['colmap_poses'] = [
                        x.tolist() for x in temp_poses_colmap
                    ]
                    debug_json[my_uuid]['new_poses'] = [
                        do_transform(tc, T_estimation, args.align).tolist()
                        for tb, tc in zip(temp_poses_base, temp_poses_colmap)
                    ]

                print(
                    f"Success at *{clip_uid}*{sidx}*{submap}*{T_estimation} with error {error}"
                )
                all_summary[clip_uid][submap] = {}
                for x in temp_colmap_invalid:
                    all_summary[clip_uid][submap][x] = do_transform(
                        temp_output[x], T_estimation,
                        args.align)[:3, :].tolist()

            all_summary[clip_uid]['common_frame_*' +
                                  submap] = temp_colmap_valid

        # this block just check if frame in colmap just appear once in every submaps
        temp_set = set()
        for sid in all_summary[clip_uid].keys():
            if 'common_frame_' not in sid and 'stat_' not in sid:
                frame_nos = list(all_summary[clip_uid][sid].keys())
                for frame_no in frame_nos:
                    if frame_no not in temp_set:
                        temp_set.add(frame_no)
                    else:
                        print(f"warning find existing frame_no {frame_no} ")

        for sid in all_summary[clip_uid].keys():
            if 'common_frame_' not in sid and 'stat_' not in sid:
                for frame_no, frame_pose in all_summary[clip_uid][sid].items():
                    if frame_no >= len(all_res[clip_uid]['good_poses']):
                        print(
                            f"overflow warning frame {frame_no} with all frames len {len(all_res[clip_uid]['good_poses'])}"
                        )
                        continue
                    if all_res[clip_uid]['good_poses'][frame_no]:
                        print(
                            f"warning want to convert colmap to baseline again for frame_no {frame_no} "
                        )
                        continue
                    assert all_res[clip_uid]['good_poses'][frame_no] is False
                    all_res[clip_uid]['good_poses'][frame_no] = True
                    all_res[clip_uid]['camera_poses'][frame_no] = frame_pose
        print(
            f"In clip {clip_uid} {clip['split']} valid/total : {len(valid_frames)}/{len(cur_base_pos['good_poses'])} submaps: {len(colmap_paths)} || {submap_log} "
        )
        # TODO : after finetune ???
        # it must be directory not .npy files
        if len(valid_frames) == 0 and clip['split'] == 'test':
            print("WARNING------------------------------------")
        print("  \n\n")

    # save the results
    # TODO: disable this for debugging
    json.dump(all_summary, open(args.new_pose, 'w'))
    json.dump(all_res, open(args.merge_pose, 'w'))
    if args.debug:
        json.dump(debug_json, open('debug.json', 'w'))
    print(f"Check output in {args.merge_pose} and {args.new_pose}")
