from genericpath import isfile
import os
import sys
import cv2
import json
import fnmatch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import json
from vq3d.bounding_box import BoundingBox

class VisualQuery3DGroundTruth():

    def __init__(self, json_file=None):
        if os.path.isfile(json_file):
            print(f"Loaded {json_file} for all poses....")
            self.all_poses = json.load(open(json_file))
        else:
            self.all_poses = None

    # NOTE : we can check here to use the json or to use other things
    # current version only supports json format pose now
    def load_pose(self, dirname: str):
        pose_dir = os.path.join(dirname, 'superglue_track', 'poses')
        clip_id = dirname.split('/')[-1]  # change -2 to -1
        original_image_ids = np.arange(
            0,
            len(
                fnmatch.filter(
                    os.listdir(dirname),  # os.path.join(dirname, 'color')
                    '*.jpg')))
        if self.all_poses:
            valid_pose = self.all_poses[clip_id]['good_poses']
            C_T_G = self.all_poses[clip_id]['camera_poses']
        else:
            raise NotImplementedError("Please always use the json pose file.")
            if not os.path.isfile(
                    os.path.join(pose_dir, 'cameras_pnp_triangulation.npy')):
                print(f"Warning : We can't find poses in {pose_dir}.")
                return None

            valid_pose = np.load(
                os.path.join(pose_dir, 'good_pose_reprojection.npy'))
            C_T_G = np.load(
                os.path.join(pose_dir, 'cameras_pnp_triangulation.npy'))

        Ci_T_G = np.zeros((len(original_image_ids), 4, 4))
        k = 0
        valid_count = 0
        # print(f"Images num: {len(original_image_ids)} Poses Num : {len(valid_pose)},{len(C_T_G)}")
        if len(original_image_ids) != len(valid_pose):
            # pass
            # padding invalid poses into
            print(
                f"Warning : Found dismatch: Images num: {len(original_image_ids)} Poses Num : {len(valid_pose)},{len(C_T_G)}."
            )
            if len(original_image_ids) > len(valid_pose):
                print("Need to pad valid pose...")
                valid_pose += [0] * (len(original_image_ids) - len(valid_pose))
        for i in range(len(original_image_ids)):
            if valid_pose[i]:
                Ci_T_G[k] = np.concatenate(
                    (C_T_G[i], np.array([[0., 0., 0., 1.]])), axis=0)
                k += 1
                valid_count += 1
            else:
                Ci_T_G[k] = np.eye(4)
                Ci_T_G[k][2, 3] = 100
                k += 1
        # if valid_count < 10:
            # print(
            #     f"Warning : in clip {clip_id} there's {valid_count} valid of {len(original_image_ids)} for camera poses. "
            # )
        return Ci_T_G, valid_pose

    def load_3d_annotation(self, data: Dict):
        # use Ego4D-3D-Annotation API
        box = BoundingBox()
        box.load(data)
        return box.center

    def create_traj_azure(self, output_traj, K, Ci_T_G=None):

        d = json.load(
            open(
                '../camera_pose_estimation/Visualization/camera_trajectory.json',
                'r'))
        dp0 = d['parameters'][0]
        dp0['intrinsic']['width'] = int((K[6] + 0.5) * 2)
        dp0['intrinsic']['height'] = int((K[7] + 0.5) * 2)
        dp0['intrinsic']['intrinsic_matrix'] = K.tolist()
        dp0['extrinsic'] = []
        x = []

        if Ci_T_G is not None:
            for i in range(Ci_T_G.shape[0]):
                temp = dp0.copy()

                # E = np.linalg.inv(G_T_Ci[i])
                E = Ci_T_G[i]

                E_v = np.concatenate([E[:, i] for i in range(4)], axis=0)
                temp['extrinsic'] = E_v.tolist()
                x.append(temp)

        d['parameters'] = x
        with open(output_traj, 'w') as f:
            json.dump(d, f)

