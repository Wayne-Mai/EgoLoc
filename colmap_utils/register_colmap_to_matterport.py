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
    
    
    parser.add_argument("--input_colmap",
                        type=str,
                        default='data/colmap_models',
                        help="Clip frames path we need poses")

    parser.add_argument("--output_dir",
                        
                        type=str,
                        default='data/colmap_aligned',
                        help="The colmap output path")

    parser.add_argument("--voc_path",
                        type=str,
                        default='data/colmap/vocab_tree_flickr100K_words1M.bin',
                        help="The colmap vocab tree path")

    parser.add_argument("--vq3d_clip_names",
                        type=str,
                        default='data/metadata/clip_names_val_test.json',
                        help="Vq3d val and test clips names")

    parser.add_argument("--camera_intrinsics",
                        type=str,
                        default='data/precomputed/scan_to_intrinsics_egoloc.json',
                        help="Vq3d scan set camera intrinsics path")

    parser.add_argument('--retry', action='store_true', default='False')

    parser.add_argument('--easy', action='store_true', default='False')

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
    my_geo_path = os.path.join('data/geo_register', my_scan_uuid)
    my_geo_align_txt = os.path.join(my_geo_path, 'geo.txt')
    my_geo_img_list = os.path.join(my_geo_path, 'image_list.txt')
    
    
    min_inlier=min_inliers[my_scan_uid]
    # debug info
    print(
        f"\n Task : {args.taskid} Clip uid : {my_uid} Scan uid : {my_scan_uid} Split : {my_split} \n"
    )

    CLIP_INPUT = os.path.join(args.input_dir, my_uid)
    CLIP_OUTPUT = os.path.join(args.output_dir, my_uid)
    colmap_output=os.path.join(args.input_colmap,my_uid)
    sparse_output = glob.glob(os.path.join(colmap_output, 'sparse/*/'))
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

    # --------------------------------------------------------------------
    # This part will do the registration
    sp.run([
        'colmap',
        "feature_extractor",
        "--database_path",
        os.path.join(colmap_output, "database.db"),
        "--image_path",
        my_geo_path,
        "--image_list_path",
        my_geo_img_list,
        "--SiftExtraction.use_gpu",
        "0",
    ],
           stderr=sys.stderr,
           stdout=sys.stdout)

    sp.run([
        'colmap',
        "vocab_tree_matcher",
        "--database_path",
        os.path.join(colmap_output, "database.db"),
        "--VocabTreeMatching.vocab_tree_path",
        args.voc_path,
        "--VocabTreeMatching.match_list_path",
        my_geo_img_list,
        "--SiftMatching.use_gpu",
        "0",
    ],
           stderr=sys.stderr,
           stdout=sys.stdout)
    # above will add new mp images to COLMAP database, which is shared by all colmap submaps
    print("Feature extraction and match finish.\n")

    # path to save colmap models transformed to mp coordinate system
    aligned_path = os.path.join(CLIP_OUTPUT, "sparse_aligned")
    if os.path.isdir(aligned_path):
        shutil.rmtree(aligned_path)
    Path(aligned_path).mkdir(parents=True, exist_ok=True)


    # path to save colmap models with both mp and ego video images
    merged_path = os.path.join(CLIP_OUTPUT, "sparse_merged")
    if os.path.isdir(merged_path):
        shutil.rmtree(merged_path)
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    
    
    
    # traverse reconstruction submaps
    for idx, so in enumerate(sparse_output):
        if os.path.isdir(os.path.join(merged_path, str(idx))):
            shutil.rmtree(os.path.join(merged_path, str(idx)))
        Path(os.path.join(merged_path, str(idx))).mkdir(parents=True, exist_ok=True)
        sp.run([
            'colmap',
            "image_registrator",
            "--database_path",
            os.path.join(colmap_output, "database.db"),
            "--input_path",
            so,
            "--output_path",
            os.path.join(merged_path, str(idx)),
            "--Mapper.init_min_num_inliers",
            min_inlier,
        ],
               stderr=sys.stderr,
               stdout=sys.stdout)
        print(f"Image registration finished.")
        
        if len(os.listdir(os.path.join(merged_path, str(idx)))) == 0:
            print(f"Warning : {idx} merge falied for {so} to {os.path.join(merged_path, str(idx))} as it's empty\n")
            continue
            

        if len(os.listdir(os.path.join(merged_path, str(idx)))) == 0:
            print(f"Warning : {idx} merge falied for {so} to {os.path.join(merged_path, str(idx))} as it's empty\n")
            continue

        if os.path.isdir(os.path.join(aligned_path, str(idx))):
            shutil.rmtree(os.path.join(aligned_path, str(idx)))
        Path(os.path.join(aligned_path, str(idx))).mkdir(parents=True, exist_ok=True)
        # above code will merge new mp images into existing colmap reconstruction
        
        sp.run([
            'colmap', 'model_aligner', '--input_path',
            os.path.join(merged_path, str(idx)), '--output_path',
            os.path.join(aligned_path, str(idx)), '--ref_images_path',
            my_geo_align_txt, '--transform_path',
            os.path.join(aligned_path, str(idx), 'sim3.txt'), '--ref_is_gps', '0',
            '--robust_alignment', '1', '--alignment_type', 'custom',
            '--estimate_scale', '1', '--robust_alignment_max_error', '5'
        ],
               stderr=sys.stderr,
               stdout=sys.stdout)
        print(f"Align finished.")
        # above code will convert merged colmap reconstruction into colmap system
        
        if len(os.listdir(os.path.join(aligned_path, str(idx)))) == 0:
            print(f"Warning : {idx} align falied for {so} to {os.path.join(aligned_path, str(idx))} as it's empty\n")
            continue
        
        sp.run([
            'colmap',
            "model_converter",
            "--input_path",
            os.path.join(aligned_path, str(idx)),
            "--output_path",
            os.path.join(aligned_path, str(idx)),
            "--output_type",
            "TXT",
        ], stderr=sys.stderr, stdout=sys.stdout)

        print(f"Success: {idx}: check aligned model in {os.path.join(aligned_path, str(idx))}")
        # now the aligned colmap reconstruction are in aligned_path
   
    print(f"ALL END AT {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
