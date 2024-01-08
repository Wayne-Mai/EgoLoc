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




"""
notes about vq3d validation set:

1. some old_annot's visual crop is different from that in vq_val,
we trust vq_val to guarantee we must have one match
2. if found>2, we use one dataset_uid prediction to match two queries, 
so you may see several warnings during validation inference
"""



def parse_vq3d_predictions(vq2d_pred, vq3d_pred):
    vq3d_pred['challenge'] = 'ego4d_vq3d_challenge'
    vq3d_pred['version'] = '1.0'
    for vidx, v in tqdm(enumerate(vq3d_pred['results']['videos'])):
        video_uid = v['video_uid']
        for cidx, c in enumerate(v['clips']):
            clip_uid = c['clip_uid']
            to_change = c['predictions']
            for ai, annot in enumerate(to_change):
                vq2d_annot = vq2d_pred[video_uid][clip_uid][ai]['query_sets']
                for qset_id, qset in annot['query_sets'].items():
                    # maybe move the test mapping here?
                    if 'query_frame_pose' not in vq2d_annot[qset_id]:
                        continue
                    else:
                        annot['query_sets'][qset_id] = {
                            'pred_3d_vec_world':
                            vq2d_annot[qset_id]['pred_3d_vec_world'],  # none
                            'pred_3d_vec_query_frame':
                            vq2d_annot[qset_id]
                            ['pred_3d_vec_query_frame'],  # none
                            'query_frame_pose':
                            vq2d_annot[qset_id]['query_frame_pose']  # none
                        }
            c['annotations'] = c.pop('predictions')
    vq3d_pred['results'] = vq3d_pred['results']['videos']
    vq3d_pred['videos'] = vq3d_pred.pop('results')
    # dictionary[new_key] = dictionary.pop(old_key)
    return vq3d_pred




def parse_VQ2D_mapper(pred_filename):

    output = {}
    data = json.load(open(pred_filename, 'r'))  # train/val_annot.json
    # added to fix, since like there's a prediction wrapped
    if 'predictions' in data.keys():
        data = data['predictions']
    if len(data['predicted_response_track']) == 1:
        # added [0] to remove, since like there's a list wrapped
        data['predicted_response_track'] = data['predicted_response_track'][0]
    for i in range(len(data['dataset_uids'])):
        dataset_uid = data['dataset_uids'][i]
        if len(data['predicted_response_track'][i]) != 1:
            print(
                f"Warning in parse_vq2d_mapper, json_parsers.py : Found len {i}:{dataset_uid} is not 1: len-{len(data['predicted_response_track'][i])}   "
            )
        if isinstance(data['predicted_response_track'][i], list):
            temp = data['predicted_response_track'][i][0]
        else:
            temp = data['predicted_response_track'][i]
        output[dataset_uid] = {
            'pred': temp,
            # * note : we shouldn't use this gt
            'gt': data['ground_truth_response_track'][i]
        }
    return output


"""
parsed_predictions:
    {  video_id:
        clip_id: [{query_set:{1:[{rt:[{xy},{xy}...];qf:;ot:;vc:;}]}}] }
"""


# ai:0
#  2: portable fan 298 3: tube 1044 1: green bowl 1498
# ai:1
#  2: stool 1499 3: bucket 363 1: stool 489
def parse_VQ2D_queries(filename: str) -> Dict:
    output = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        video_uid = video['video_uid']
        if video_uid not in output:
            output[video_uid] = {}
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            if clip_uid not in output[video_uid]:
                output[video_uid][clip_uid] = []
            for ai, annot in enumerate(
                    clip['annotations']):  # traverse a list [{'query_set'}...]
                annot_handle = {
                    'ai_list': ai,
                    # "annotation_uid": annot["annotation_uid"],
                    'query_sets': {}
                }
                # * note : validation set inconsistency here, v1_0_5 has the annotation_uid
                if 'annotation_uid' in annot:
                    annot_handle["annotation_uid"] = annot["annotation_uid"]

                for qset_id, qset in annot['query_sets'].items():
                    if not qset["is_valid"]:
                        annot_handle['query_sets'][qset_id] = {}
                        continue
                    annot_handle['query_sets'][qset_id] = {
                        "query_frame": qset["query_frame"],
                        # "object_title": qset["object_title"],
                        "visual_crop": qset["visual_crop"],
                        # "response_track": qset["response_track"],
                    }
                    if 'response_track' in qset:
                        annot_handle['query_sets'][qset_id][
                            'response_track'] = qset['response_track']
                    if 'object_title' in qset:
                        annot_handle['query_sets'][qset_id][
                            'object_title'] = qset['object_title']

                output[video_uid][clip_uid].append(annot_handle)
    return output


def parse_VQ2D_queries_reorganize(filename: str) -> Dict:
    output = {}
    output_header = {}
    data = json.load(open(filename, 'r'))
    for video in data['videos']:
        video_uid = video['video_uid']
        if video_uid not in output:
            output[video_uid] = {}
            output_header[video_uid] = {}
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            if clip_uid not in output[video_uid]:
                output[video_uid][clip_uid] = []
                output_header[video_uid][clip_uid] = {}
            for k in clip.keys():
                if k != 'annotations':
                    output_header[video_uid][clip_uid][k] = clip[k]
            for ai, annot in enumerate(
                    clip['annotations']):  # traverse a list [{'query_set'}...]
                annot_handle = {
                    'ai_list': ai,
                    # "annotation_uid": annot["annotation_uid"],
                    'query_sets': {}
                }
                # * note : validation set inconsistency here, v1_0_5 has the annotation_uid
                if 'annotation_uid' in annot:
                    annot_handle["annotation_uid"] = annot["annotation_uid"]

                for qset_id, qset in annot['query_sets'].items():
                    if not qset["is_valid"]:
                        annot_handle['query_sets'][qset_id] = {}
                        continue
                    annot_handle['query_sets'][qset_id] = {
                        "query_frame": qset["query_frame"],
                        # "object_title": qset["object_title"],
                        "visual_crop": qset["visual_crop"],
                        # "response_track": qset["response_track"],
                    }
                    if 'response_track' in qset:
                        annot_handle['query_sets'][qset_id][
                            'response_track'] = qset['response_track']
                    if 'object_title' in qset:
                        annot_handle['query_sets'][qset_id][
                            'object_title'] = qset['object_title']

                output[video_uid][clip_uid].append(annot_handle)
    return output, output_header


"""for val format
    parsed_predictions:
     {  video_id:
            clip_id: [{query_set:{1:{object_title:...;pred_response_track:
            {bbox:[{xy},{xy}...]}]
            }} }
"""


def parse_VQ2D_predictions_train_val(pred_filename, annot_filename,
                                     gt_vq2d_queries):
    vq2d_pred = parse_VQ2D_mapper(pred_filename)
    # print(vq2d_pred.keys())
    vq2d_queries = copy.deepcopy(gt_vq2d_queries)
    data = json.load(open(annot_filename, 'r'))
    output = {}

    for i in range(len(data)):
        dataset_uid = data[i]['dataset_uid']
        video_uid = data[i]['metadata']['video_uid']
        clip_uid = data[i]['clip_uid']
        query_set = data[i]['query_set']
        query_frame = data[i]['query_frame']
        object_title = data[i]['object_title']
        visual_crop = data[i]['visual_crop']

        # associate the dataset_uid to the ai, query_set_id
        found = 0
        for ai, annot in enumerate(vq2d_queries[video_uid][clip_uid]):
            for qset_id, qset in annot['query_sets'].items():
                if not qset: continue
                if qset['query_frame'] == query_frame and qset_id == query_set and qset[
                        'object_title'] == object_title \
                            and check_crop_equal(qset['visual_crop'],visual_crop):
                    if 'old_annot' in qset:
                        print(
                            f"Warning in json_parsers.py: annot {dataset_uid} map to both {qset['old_annot']['dataset_uid']} "
                        )
                    qset['old_annot'] = data[i]

                    # put our vq2d prediction here
                    if dataset_uid in vq2d_pred:
                        qset['pred_response_track'] = vq2d_pred[dataset_uid][
                            'pred']
                    else:
                        qset['pred_response_track'] = {}
                        print(
                            f"Warning in json_parsers.py: annot {dataset_uid} not found in vq2d results "
                        )
                    found += 1
        if found != 1:
            print(
                f"Warning in json_parsers: Found {found} in annot {dataset_uid}."
            )
    return vq2d_queries


"""for test format
    parsed_predictions:
     {  video_id:
            clip_id: [{query_set:{1:[{bbox:[{xy},{xy}...]}]}}] }
"""


def parse_VQ2D_predictions(filename: str):
    output = {}
    data = json.load(open(filename, 'r'))
    vq3d_pred = copy.deepcopy(data)
    data = data['results']
    for vidx, v in tqdm(enumerate(data['videos'])):
        if v['video_uid'] not in output:
            output[v['video_uid']] = {}
        for cidx, c in enumerate(v['clips']):
            if c['clip_uid'] not in output[v['video_uid']]:
                output[v['video_uid']][c['clip_uid']] = {}
            output[v['video_uid']][c['clip_uid']] = c['predictions']

            to_change = vq3d_pred['results']['videos'][vidx]['clips'][cidx][
                'predictions']
            for ai, annot in enumerate(to_change):
                for qset_id, qset in annot['query_sets'].items():
                    annot['query_sets'][qset_id] = {
                        'pred_3d_vec_world': None,  # none
                        'pred_3d_vec_query_frame': None,  # none
                        'query_frame_pose': None  # none
                    }
    return output, vq3d_pred


def check_crop_equal(crop1, crop2):
    return crop1['frame_number'] == crop2['frame_number']

    # sometimes video_frame_number can +- 1
    for k in crop1:
        # print(f"{k} {crop1[k]} {crop2[k]} {crop1[k] == crop2[k]}")
        if crop1[k] != crop2[k]:
            if k == 'video_frame_number' and abs(crop1[k] - crop2[k]) <= 2:
                continue
            print(crop1)
            print(crop2)
            print('false')
            return False

    # print('true')
    return True



def reformat_vq3d_test_v2(vq3d_pred):
    for video in vq3d_pred['videos']:
        for clip in video['clips']:
            for ai, annot in enumerate(clip['annotations']):
                if not annot: continue
                for qset_id, qset in annot['query_sets'].items():
                    annot['query_sets'][qset_id] = {
                        'pred_3d_vec_world':
                        qset['mp_pred_3d_vec_world']
                        if 'pred_3d_vec_world' in qset else None,
                        'pred_3d_vec_query_frame':
                        qset['mp_pred_3d_vec_query_frame']
                        if 'rotated_pred_3d_vec_query_frame' in qset
                        and qset['rotated_pred_3d_vec_query_frame'] else None,
                        'query_frame_pose': 
                        np.array(
                            qset['query_frame_pose']).tolist()
                        if 'query_frame_pose' in qset
                        and qset['query_frame_pose'] else None,
                    }
    return vq3d_pred

