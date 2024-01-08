import os
import sys
import json
import h5py
import argparse
import numpy as np
from PIL import Image
import copy
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from vq3d.get_query_3d_ground_truth import VisualQuery3DGroundTruth
from vq3d.json_parser import parse_VQ2D_queries,parse_VQ2D_predictions,parse_VQ2D_predictions_train_val,reformat_vq3d_test_v2
from vq3d.utils import  format_frames,scale_im_height
from vq3d.filter import frame_filter,camera_filter
from vq3d.mv_aggregate import predict_for_frame_j,multi_frames_prediction
from vq3d.depth_utils import DPT_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        default='data/clips_frames',
                        help="Location of extracted video clip frames")

    parser.add_argument("--pose_dir",
                        type=str,
                        default='data/all_clips_base_colmap_v2.json',
                        help="Camera pose json file")

    parser.add_argument(
        "--output_filename",
        type=str,
        default='data/vq3d_results/siam_rcnn_residual_kys_val.json',
        help="Saved prediction file")
    parser.add_argument(
        "--vq2d_results",
        type=str,
        default='data/baseline_test_challenge_predictions.json',
        help="VQ2D results")
    parser.add_argument("--vq2d_annot",
                        type=str,
                        default='data/metadata/val_annot.json',
                        help="VQ2D annotations ")
    parser.add_argument("--vq3d_queries",
                        type=str,
                        default='data/vq3d_test_unannotated.json',
                        help="VQ3D query file")
    parser.add_argument("--vq2d_queries",
                        type=str,
                        default='data/v1/annotations/vq_test_unannotated.json',
                        help="VQ3D query file")
    parser.add_argument(
        "--vq_mappings",
        type=str,
        default='data/metadata/mapping_vq2d_to_vq3d_queries_annotations_test.json',
        help="VQ3D-2D mapping file")
    parser.add_argument("--use_gt", action='store_true')
    parser.add_argument("--use_depth_from_scan", action='store_true')
    parser.add_argument("--camera_intrinsics",
                        type=str,
                        default='data/precomputed/scan_to_intrinsics_egoloc.json',
                        help="Vq3d scan set camera intrinsics path")
    parser.add_argument("--vq3d_clip_names",
                        type=str,
                        default='data/metadata/clip_names_val_test.json',
                        help="Vq3d val and test clips names")
    parser.add_argument("--experiment",
                        type=str,
                        default='',
                        help="our experiment identifier")
    parser.add_argument(
        "--mode",
        type=str,
        default='test',
        help="We support validation and test set")
    
    # usually you don't need to care these two arguments, we won't use them by default
    # args.nms_threshold=0.3 spatial l2 distance in meters, or 10000
    # args.nms_fusion=True: merge the observation if min>0.5*max clip score
    # args.nms_fusion_threshold: <= 0 means no fusion, always return 
    # top 1>0, means we use weighted average
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.5,
        help="Merge points according to distance, only valid when nms is in experiment")
    
    parser.add_argument(
        "--nms_fusion_threshold",
        type=float,
        default=-1,
        help="Merge close points according to score, only valid when nms is in experiment")
    
    
    args = parser.parse_args()

    # todo maybe remove this preset and make it decide by args ?
    if args.mode == 'test':
        args.vq2d_queries = 'data/metadata/vq_test_unannotated.json'
        args.vq3d_queries = 'data/metadata/vq3d_test_unannotated_template.json' # is this new ?
        args.vq_mappings = 'data/metadata/mapping_vq2d_to_vq3d_queries_annotations_test.json'
        # args.vq3d_clip_names = 'data/v1/3d/all_clips_for_vq3d_wayne_test.json'
    elif args.mode == 'val':
        # * note if we use v1_0_5, weird things will happen with ai/qset_id dismatch
        # disable this as we are using different format of vq2d val queries results 
        # args.vq2d_queries = 'data/v1/annotations/vq_val.json'
        args.vq3d_queries = 'data/metadata/vq3d_val.json'
        args.vq_mappings = 'data/metadata/mapping_vq2d_to_vq3d_queries_annotations_val.json'
        # args.vq2d_annot = 'data/metadata/val_annot.json'
    # elif args.mode == 'train': # todo: support train set later
    #     args.vq2d_queries = 'data/v1/annotations/vq_train.json'
    #     args.vq3d_queries = 'data/v1/3d/vq3d_train.json'
    #     args.vq_mappings = 'data/mapping_vq2d_to_vq3d_queries_annotations_train.json'
    #     args.vq2d_annot = 'data/train_annot.json'

    else:
        raise Exception("Mode not found")

    args.output_filename = args.output_filename.split('.')
    args.output_filename = args.output_filename[
        0] + '_' + args.experiment + '_' + args.mode + '.' + args.output_filename[1]

    # experiment intrinsics handler
    with open(args.camera_intrinsics, 'r') as f:
        cam_k = json.load(f)
    with open(args.vq3d_clip_names, 'r') as f:
        clip_uids = json.load(f)

    root_dir = args.input_dir
    all_query_count,no_pose_count,no_rt_pose_count=0,0,0
    output_filename = args.output_filename

    # Visual Query 3D queries
    vq3d_queries = json.load(open(args.vq3d_queries, 'r'))

    # Visual Query 2D results
    vq2d_queries = parse_VQ2D_queries(args.vq2d_queries)
    if args.mode == 'test':
        vq2d_pred, _ = parse_VQ2D_predictions(args.vq2d_results)
    else:
        vq2d_pred = parse_VQ2D_predictions_train_val(args.vq2d_results,
                                                     args.vq2d_annot,
                                                     vq2d_queries)
   
    query_matching = json.load(open(args.vq_mappings, 'r'))

    helper = VisualQuery3DGroundTruth(args.pose_dir)

    for video in tqdm(vq3d_queries['videos']):
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']

            # * experiment noted : add camera intrinsics
            #--------------------------------------------------------------
            CLIP_INPUT = os.path.join(args.input_dir, clip_uid)
            # We need to check the resolution to select the correct intrinsics
            example_img = Image.open(
                os.path.join(CLIP_INPUT, 'color_0000000.jpg'))
            resolution = example_img.size  # (width, height)
            resolution_token = (str(resolution[0]), str(resolution[1]))
            resolution_token = str(resolution_token)
            example_img.close()
            my_scan_uid = None
            for clip_dct in clip_uids['clips']:
                if clip_dct['clip_uid'] == clip_uid:
                    my_scan_uid = clip_dct['scan_uid']
            my_camera_k = cam_k[my_scan_uid]
            assert my_scan_uid is not None
            CAM_PARAS = ""
            paras = ['f', 'cx', 'cy', 'k1', 'k2']
            paras_value = []
            for p in paras:
                paras_value.append(float(my_camera_k[resolution_token][p]))
            f, cx, cy, k1, k2 = paras_value
            W, H = cx * 2.0, cy * 2.0
            #--------------------------------------------------------------

            # video_uid, clip_uid is known now
            for ai, annot in enumerate(clip['annotations']):
                if not annot: continue
                # * note : we can put following code outside of this loop maybe 
                dirname = os.path.join(root_dir, clip_uid)  #, 'egovideo')
                if not os.path.isdir(dirname): 
                    print(f"Warning : Jump {dirname} as it doesn't exist")
                    continue
                T, valid_pose = helper.load_pose(dirname)
                cur_query_count=len(annot['query_sets'].keys())
                all_query_count+=cur_query_count
                # optional
                # if not any(valid_pose):
                #     no_pose_count+=cur_query_count 
                #     print(f"Error : clip {clip_uid} completely failed")
                #     continue
                
                
                
                for qset_id, qset in annot['query_sets'].items():
                    # get corresponding vq2d_query as well
                    # note we need to convert 3d ai/qset_id to 2d
                    mapping_ai = query_matching[video_uid][clip_uid][str(
                        ai)][qset_id]['ai']
                    mapping_qset_id = query_matching[video_uid][clip_uid][str(
                        ai)][qset_id]['qset_id']
                    cur_vq2d_query_qset = vq2d_queries[video_uid][clip_uid][
                        mapping_ai]['query_sets'][mapping_qset_id]
                    cur_vq2d_pred_qset = vq2d_pred[video_uid][clip_uid][
                        mapping_ai]['query_sets'][mapping_qset_id]

                    query_frame = cur_vq2d_query_qset['query_frame']
                    oW = cur_vq2d_query_qset["visual_crop"]["original_width"]
                    oH = cur_vq2d_query_qset["visual_crop"]["original_height"]
                    object_title = cur_vq2d_query_qset['object_title']
                    # consistence check, 2d_query_title==2d_pred_title, 2d_query_title==3d_query_title
                    if 'object_title' in cur_vq2d_pred_qset:
                        assert object_title == cur_vq2d_pred_qset['object_title']
                    
                    if 'object_title' in qset and object_title != qset['object_title']:
                        print(
                        f"Warning in 3d-2d object title query matching: clip:{clip_uid} ai:{ai} qset_id:{qset_id}"
                    )
                    # prepare the groundtruth for validation set
                    center_vectors = []
                    if args.mode != 'test':
                        # assert object_title==qset['object_title']
                        # print("success")
                        for w in [1, 2]:
                            center_vec = helper.load_3d_annotation(
                                qset[f'3d_annotation_{w}'])
                            # only the center location is returned
                            qset[f'gt_3d_vec_world_{w}'] = center_vec.tolist()
                            qset[f'mp_gt_3d_vec_world_{w}'] = [
                                center_vec[1], -center_vec[0], center_vec[2]
                            ]
                            center_vectors.append(
                                np.array(qset[f'mp_gt_3d_vec_world_{w}']))
                        

                    # frame_indices_valid: global frame index in the entire video clip
                    # local_frame_indices: frame index in detected response track
                    frame_indices_valid = []
                    local_frame_indices = []

                    if args.mode!='test':
                        cur_vq2d_query_qset['response_track'] = format_frames(
                            cur_vq2d_query_qset['response_track'])
                    # if args.mode != 'test':
                    if 'usegt' in args.experiment:
                        response_track = cur_vq2d_query_qset[
                            'response_track']
                        frames = response_track
                        # print(frames)
                    else:
                        if args.mode!='test':
                            response_track = cur_vq2d_pred_qset[
                                'pred_response_track']
                        else:
                            response_track=cur_vq2d_pred_qset
                        frames = response_track['bboxes']
                        
                        frames=frame_filter(args, frames)

                    

                    frame_indices = [x['fno'] for x in frames]

                    DPT_process(frame_indices, os.path.abspath(dirname))

                    for i, frame_index in enumerate(frame_indices):

                        # check if frame index is valid
                        if (frame_index > -1) and (frame_index <
                                                   len(valid_pose)):

                            # check if box is within frame bound:
                            box = frames[i]
                            x1, x2, y1, y2 = box['x1'], box['x2'], box[
                                'y1'], box['y2']

                            if (x1 <
                                (W - 1)) and (x2 > 1) and (y1 <
                                                           (H - 1)) and (y2 >
                                                                         1):

                                # check if pose is valid
                                if valid_pose[frame_index]:
                                    frame_indices_valid.append(frame_index)
                                    local_frame_indices.append(i)

                    if len(frame_indices_valid) == 0: 
                        no_rt_pose_count+=1
                        continue
                    
                    frame_indices_valid=camera_filter(args,frame_indices_valid,T)

                    # check if Query frame has pose
                    if valid_pose[query_frame]:
                        pose_Q = T[query_frame]
                        qset['query_frame_pose'] = pose_Q.tolist()
                    else:
                        pose_Q = None
                        qset['query_frame_pose'] = None

                    depth_dir = os.path.join(
                        root_dir,
                        clip_uid,
                        #  'egovideo',
                        'depth_DPT_predRT')

                    if args.mode != 'test':
                        qset['gt_response_track'] = cur_vq2d_query_qset[
                            'response_track']
                        qset['camera_intrinsics'] = my_camera_k[
                            resolution_token]
                        qset['scan_uid'] = my_scan_uid

                    if 'last' in args.experiment:
                        j = np.argmax(frame_indices_valid)
                        res = predict_for_frame_j(args, frame_indices_valid,
                                                  local_frame_indices, frames,
                                                  T, pose_Q, paras_value,
                                                  depth_dir, center_vectors, j)
                    else: # using multiview aggregation
                        all_res = []
                        for j in range(len(frame_indices_valid)):
                            temp_res = predict_for_frame_j(
                                args, frame_indices_valid, local_frame_indices,
                                frames, T, pose_Q, paras_value, depth_dir,
                                center_vectors, j)
                            if temp_res:
                                all_res.append(temp_res)
                        res = multi_frames_prediction(args, all_res)
                    for k, v in res.items():
                        qset[k] = v


   
    vq3d_pred = vq3d_queries
    if args.mode =='test':
        vq3d_pred=reformat_vq3d_test_v2(vq3d_pred)
    
    
    json.dump(vq3d_pred, open(output_filename, 'w'))
    print(f"Run Summary: All query : {all_query_count} Query w/o registration {no_pose_count} Query w/o pose frame {no_rt_pose_count} ")
    print(f"Check result file in {output_filename}")