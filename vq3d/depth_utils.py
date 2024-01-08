
import copy
import os, json, sys
import shutil
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path
import subprocess as sp
import pathlib
import glob
import numpy as np
import torch
import sys
import re
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from vq3d.utils import  _get_box


def read_pfm(path):
    """Read pfm file.
    Args:
        path (str): path to file
    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$",
                             file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale
    
    

# TODO: currently we use distorted image, we should try undistorted image afterwards
def DPT_process(frame_indices, parent_dir, undistorted=False):
    tmp_input_depth_dir = os.path.join(parent_dir, 'input_depth_DPT_predRT')
    tmp_output_depth_dir = os.path.join(parent_dir, 'depth_DPT_predRT')

    Path(tmp_input_depth_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_output_depth_dir).mkdir(parents=True, exist_ok=True)

    need_DPT = False
    # processed_files=glob.glob(tmp_output_depth_dir+'/*.pfm')

    for frame_index in frame_indices:
        framename = 'color_%07d' % frame_index
        temp_image_path = os.path.join(parent_dir, framename + '.jpg')
        if not os.path.isfile(temp_image_path):
            print(
                f"Warning in DPT: {temp_image_path} missing... Re-download...")
            raise NotImplementedError("SFTP Implemnetation")
        if os.path.isfile(
                os.path.join(tmp_output_depth_dir, framename + '.pfm')):
            continue
        else:
            shutil.copy(temp_image_path,
                        os.path.join(tmp_input_depth_dir, framename + '.jpg'))
            need_DPT = True

    if not need_DPT:
        return
    print(f"Warning in DPT: {parent_dir} need to be processed...")

    if undistorted:
        # TODO : need to implement this for large k1 and k2
        # do something to get undistored version of images
        pass

    file_abs_path = pathlib.Path(__file__).parent.resolve()
    dpt_path = file_abs_path.parents[0] / 'DPT'
    # pathlib.Path().resolve()
    # print(dpt_path)
    print(tmp_input_depth_dir)
    # TODO: merge this command as internal function
    sp.run([
        'python', 'run_monodepth.py', '-t', 'dpt_hybrid_nyu', '-i',
        tmp_input_depth_dir, '-o', tmp_output_depth_dir
    ],
           stderr=sys.stderr,
           stdout=sys.stdout,
           cwd=dpt_path)
    


def get_object_depth(depth_dir,
                     frame_index_valid,
                     local_frame_bbox,
                     H,
                     W,
                     use_gt=False):
    # get depth
    # depth_dir = os.path.join(
    #     root_dir,
    #     clip_uid,
    #     #  'egovideo',
    #     'depth_DPT_predRT')
    framename = 'color_%07d' % frame_index_valid
    depth_filename = os.path.join(depth_dir, framename + '.pfm')
    if os.path.isfile(depth_filename):
        data, scale = read_pfm(depth_filename)
    else:
        print('missing predicted depth! Should never happen', depth_filename)
        raise Exception("NO DEPTH")
        # continue

    depth = data / 1000.0  # in meters

    # resize depth
    depth = torch.FloatTensor(depth)
    depth = depth.unsqueeze(0).unsqueeze(0)
    if use_gt:
        depth = torch.nn.functional.interpolate(depth,
                                                size=(int(oH), int(oW)),
                                                mode='bilinear',
                                                align_corners=True)
    else:
        depth = torch.nn.functional.interpolate(depth,
                                                size=(int(H), int(W)),
                                                mode='bilinear',
                                                align_corners=True)
    depth = depth[0][0].cpu().numpy()

    # select d
    if use_gt:
        # we don't have gt, this will never be called
        box = _get_box(response_track[local_frame_index])
        x1, y1, x2, y2 = box
    else:
        # box = frames[local_frame_index]
        box = local_frame_bbox
        x1 = box['x1']
        x2 = box['x2']
        y1 = box['y1']
        y2 = box['y2']
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
    # print(x1,x2,y1,y2)
    d = depth[y1:y2, x1:x2]

    return d, x1, x2, y1, y2
