from genericpath import isdir
import os
import sys
import json
import argparse
import subprocess as sp
from PIL import Image
from pathlib import Path
from datetime import datetime
import shutil
import glob

if __name__ == '__main__':
    print(f"START AT {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser()

    parser.add_argument("--taskid",
                        type=int,
                        default=0,
                        help="Slurm task id to decide clip idx to process")

    parser.add_argument("--input_dir",
                        type=str,
                        default='data/clips_frames',
                        help="Clip frames path we need poses")

    parser.add_argument("--output_dir",
                        type=str,
                        default='data/colmap_models',
                        help="The colmap output path")

    parser.add_argument("--voc_path",
                        type=str,
                        default='data/colmap/vocab_tree_flickr100K_words1M.bin',
                        help="The colmap vocab tree path, you can download it from COLMAP's official repo or our shared google drive")

    parser.add_argument("--vq3d_clip_names",
                        type=str,
                        default='data/metadata/clip_names_val_test.json',
                        help="Vq3d val and test clips names")

    parser.add_argument("--camera_intrinsics",
                        type=str,
                        default='data/precomputed/scan_to_intrinsics.json',
                        help="Vq3d scan set camera intrinsics path")


    args = parser.parse_args()
    # add scan_uid to the all_clips_for_vq3d_v1.json
    # check

    # load the jsons we need
    with open(args.vq3d_clip_names, 'r') as f:
        clip_uids = json.load(f)
    with open(args.camera_intrinsics, 'r') as f:
        cam_k = json.load(f)

    # {
    #     "clips": [
    #         {
    #             "clip_uid": "6c641082-044e-46a7-ad5f-85568119e09e",
    #             "split": "val",
    #             "scan_uid": "unict_Scooter mechanic_31"
    #         },

    my_uid = clip_uids['clips'][args.taskid]['clip_uid']
    my_scan_uid = clip_uids['clips'][args.taskid]['scan_uid']
    my_split = clip_uids['clips'][args.taskid]['split']
    my_camera_k = cam_k[my_scan_uid]
    scan_name_to_uid = {
        'unict_Scooter mechanic_31': 'unict_3dscan_001',
        'unict_Baker_32': 'unict_3dscan_002',
        'unict_Carpenter_33': 'unict_3dscan_003',
        'unict_Bike mechanic_34': 'unict_3dscan_004',
    }
    min_inliers={
        'unict_Scooter mechanic_31': '50',
        'unict_Baker_32': '20',
        'unict_Carpenter_33': '20',
        'unict_Bike mechanic_34': '50',
    }
    my_scan_uuid = scan_name_to_uid[my_scan_uid]
  
    min_inlier=min_inliers[my_scan_uid]
    # debug info
    print(
        f"\n Task : {args.taskid} Clip uid : {my_uid} Scan uid : {my_scan_uid} Split : {my_split} \n"
    )

    CLIP_INPUT = os.path.join(args.input_dir, my_uid)
    CLIP_OUTPUT = os.path.join(args.output_dir, my_uid)
    CAM_MODEL = "RADIAL_FISHEYE"

    # We need to check the resolution to select the correct intrinsics
    example_img = Image.open(os.path.join(CLIP_INPUT, 'color_0000000.jpg'))
    resolution = example_img.size  # (width, height)
    resolution_token = (str(resolution[0]), str(resolution[1]))
    resolution_token = str(resolution_token)
    example_img.close()
    CAM_PARAS = ""
    paras = ['f', 'cx', 'cy', 'k1', 'k2']
    for p in paras:
        CAM_PARAS += str(my_camera_k[resolution_token][p])
        if p != 'k2':
            CAM_PARAS += ','

    print(
        f"\n CLIP_INPUT {CLIP_INPUT} CLIP_OUTPUT {CLIP_OUTPUT} CAM_PARAS {CAM_PARAS}  \n"
    )


    Path(CLIP_OUTPUT).mkdir(parents=True, exist_ok=True)
    o = sp.check_output([
        'colmap',
        "feature_extractor",
        "--database_path",
        os.path.join(CLIP_OUTPUT, "database.db"),
        "--image_path",
        CLIP_INPUT,
        "--ImageReader.camera_model",
        CAM_MODEL,
        "--ImageReader.camera_params",
        CAM_PARAS,
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.num_threads",
        "16",
        "--SiftExtraction.use_gpu",
        "0",
    ])
    print("Feature extraction finish.\n",o)

    o = sp.check_output([
        'colmap',
        "sequential_matcher",
        "--database_path",
        os.path.join(CLIP_OUTPUT, "database.db"),
        "--SequentialMatching.vocab_tree_path",
        args.voc_path,
        "--SequentialMatching.loop_detection",
        "1",
        "--SiftMatching.num_threads",
        "16",
        "--SiftMatching.use_gpu",
        "0",
    ])
    print("Feature match finish.\n",o)

    Path(os.path.join(CLIP_OUTPUT, "sparse")).mkdir(parents=True, exist_ok=True)
    sp.run([
        'colmap',
        "mapper",
        "--database_path",
        os.path.join(CLIP_OUTPUT, "database.db"),
        "--image_path",
        CLIP_INPUT,
        "--output_path",
        os.path.join(CLIP_OUTPUT, "sparse"),
    ], stderr=sys.stderr, stdout=sys.stdout)
    print("Mapping finish.\n")

    sparse_output = glob.glob(os.path.join(CLIP_OUTPUT, 'sparse/*/'))

    print(f"We find {len(sparse_output)} submaps in this clip.")
    for so in sparse_output:
        sp.run([
            'colmap',
            "model_converter",
            "--input_path",
            so,
            "--output_path",
            so,
            "--output_type",
            "TXT",
        ], stderr=sys.stderr, stdout=sys.stdout)

    print(f"ALL END AT {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
