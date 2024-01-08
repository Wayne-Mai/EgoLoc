import gzip
import json
import multiprocessing as mp
import os
import os.path as osp
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pims
import skimage.io
import torch
import tqdm
from detectron2.utils.logger import setup_logger
from detectron2_extensions.config import get_cfg as get_detectron_cfg
from scipy.signal import find_peaks, medfilt
from vq2d.baselines import (
    create_similarity_network,
    get_clip_name_from_clip_uid,
    perform_retrieval,
    SiamPredictor,
)
from vq2d.structures import ResponseTrack
from vq2d.tracking import Tracker

setup_logger()

import hydra
from omegaconf import DictConfig, OmegaConf
import pickle

# in validation set, every single object title will have a dataset uid
# in test set, every 3 object title


def get_images_at_peak(all_bboxes, all_scores, all_imgs, peak_idx, topk=5):
    bboxes = all_bboxes[peak_idx]
    scores = all_scores[peak_idx]
    image = all_imgs[peak_idx]
    # Visualize the top K retrievals from peak image
    bbox_images = []
    for bbox in bboxes[:topk]:
        bbox_images.append(image[bbox.y1:bbox.y2 + 1, bbox.x1:bbox.x2 + 1])
    return bbox_images


def predict_vq_test(annotations, cfg, device_id, use_tqdm=False):

    data_cfg = cfg.data
    

    dataset_predictions = []

    device = torch.device(f"cuda:{device_id}")

    # Create detector
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.merge_from_file(cfg.model.config_path)
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.model.score_thresh
    detectron_cfg.MODEL.WEIGHTS = cfg.model.checkpoint_path
    detectron_cfg.MODEL.DEVICE = f"cuda:{device_id}"
    detectron_cfg.INPUT.FORMAT = "RGB"
    predictor = SiamPredictor(detectron_cfg)

    # Create tracker
    similarity_net = create_similarity_network()
    similarity_net.eval()
    similarity_net.to(device)

    # Visualization
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    if cfg.logging.visualize:
        OmegaConf.save(cfg, os.path.join(cfg.logging.save_dir, "config.yaml"))

    annotations_iter = tqdm.tqdm(annotations) if use_tqdm else annotations
    for annotation in annotations_iter:
        start_time = time.time()
        clip_uid = annotation["clip_uid"]
        print(f"I'm tracking {clip_uid} on device {device_id} ")
        annotation_uid = annotation["metadata"]["annotation_uid"]
        # Load clip from file
        clip_path = os.path.join(data_cfg.data_root,
                                 get_clip_name_from_clip_uid(clip_uid))
        video_reader = pims.Video(clip_path)
        query_frame = annotation["query_frame"]
        visual_crop = annotation["visual_crop"]
        vcfno = annotation["visual_crop"]["frame_number"]
        clip_frames = video_reader[0:max(query_frame, vcfno) + 1]
        clip_read_time = time.time() - start_time
        start_time = time.time()
        # Retrieve nearest matches and their scores per image
        ret_bboxes, ret_scores, ret_imgs, visual_crop_im = perform_retrieval(
            clip_frames,
            visual_crop,
            query_frame,
            predictor,
            batch_size=data_cfg.rcnn_batch_size,
            recency_factor=cfg.model.recency_factor,
            subsampling_factor=cfg.model.subsampling_factor,
        )
        detection_time_taken = time.time() - start_time

        print("====> Data uid: {} | search window :{:>8d} frames | "
              "clip read time: {:>6.2f} mins | "
              "detection time: {:>6.2f} mins | ".format(
                  annotation["clip_uid"],
                  annotation["query_frame"],
                  clip_read_time / 60.0,
                  detection_time_taken / 60.0,
              ))

        # * note : code from frost, save the predictions
        ret_bboxes = [[bbox.to_json() for bbox in ret_bbox] for ret_bbox in ret_bboxes]
        # {'metadata': {'video_uid': '17607d8a-806f-4990-b5e7-756b6389c826', 'video_start_sec': 0.0,
        #               'video_end_sec': 480.0, 'clip_fps': 5.0, 'query_set': '2', 
        #               'annotation_uid': '09109295-5bc4-4f62-ba44-16e6f6c5301d'}, '
        #  clip_uid': '634a5fae-5d81-460e-b82b-b5aaa4d77cb3', 
        #  'query_frame': 393, 'visual_crop': {'frame_number': 173, 'x': 1727.8, 'y': 358.99, 
        # 'width': 104.03, 'height': 63.14, 'rotation': 0, 'original_width': 1920, 
        # 'original_height': 1080, 'video_frame_number': 1038}}
        prediction = {
            'top_boxes_per_clip': ret_bboxes, # 1081, [bbox_obj_fno_0, bbox_fno_0, .....]
            'top_scores_per_clip':ret_scores, # 1081, [[score, score, .....]]color_00000
            'clip_uid':annotation['clip_uid'],
            'query_frame':annotation['query_frame'],
            'visual_crop':annotation['visual_crop'],
            'visual_crop_fno':annotation['visual_crop']['frame_number'],
            'metadata': annotation['metadata']
        }
        dataset_predictions.append(prediction)
        # dataset_uids.append(annotation["dataset_uid"])




    return dataset_predictions


def _mp_aux_fn(inputs):
    return predict_vq_test(*inputs)


def predict_vq_test_parallel(annotations, cfg):
    if cfg.data.debug_mode:
        cfg.data.num_processes = 1

    context = mp.get_context("forkserver")
    pool = context.Pool(cfg.data.num_processes, maxtasksperchild=2)
    # Split data across processes
    B = cfg.data.batch_size
    mp_annotations = [
        annotations[i:(i + B)] for i in range(0, len(annotations), B)
    ]
    N = len(mp_annotations)
    devices = [i for i in range(torch.cuda.device_count())]
    mp_cfgs = [cfg for _ in range(N)]
    mp_devices = [devices[i % len(devices)] for i in range(N)]
    mp_inputs = zip(mp_annotations, mp_cfgs, mp_devices)
    # Perform task
    list_of_outputs = list(tqdm.tqdm(pool.imap(_mp_aux_fn, mp_inputs),
                                     total=N))
    predicted_rts = []
    for output in list_of_outputs:
        predicted_rts += output
    return predicted_rts


                        
def convert_annotations_to_list(annotations):
    annotations_list = []
    for v in annotations["videos"]:
        vuid = v["video_uid"]
        for c in v["clips"]:
            cuid = c["clip_uid"]
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    if not q['is_valid']:
                        continue
                    annotations_list.append({
                        "metadata": {
                            "video_uid": vuid,
                            "video_start_sec": c["video_start_sec"],
                            "video_end_sec": c["video_end_sec"],
                            "clip_fps": c["clip_fps"],
                            "query_set": qid,
                            "annotation_uid": aid,
                        },
                        "clip_uid": cuid,
                        "query_frame": q["query_frame"],
                        "visual_crop": q["visual_crop"],
                    })
    return annotations_list


# "metadata": {
#                 "video_uid": vuid,
#                 "video_start_sec": c["video_start_sec"],
#                 "video_end_sec": c["video_end_sec"],
#                 "clip_fps": c["clip_fps"],
#                 "query_set": qid,
#                 "annotation_uid": aid,
#             },
#             "clip_uid": cuid,
#             "query_frame": q["query_frame"],
#             "visual_crop": q["visual_crop"],



def format_predictions(annotations,
                       annotations_list,
                       predicted_rts,
                       debug=False):
    # Get predictions for each annotation_uid
    annotation_uid_to_prediction = {}
    for pred_rt, annot in zip(predicted_rts, annotations_list):
        auid = annot["metadata"]["annotation_uid"]
        query_set = annot["metadata"]["query_set"]
        if auid not in annotation_uid_to_prediction:
            annotation_uid_to_prediction[auid] = {}
        annotation_uid_to_prediction[auid][query_set] = [
            rt.to_json() for rt in pred_rt
        ]

    if debug:
        print(
            f"Formatting the json file: \n\n {annotation_uid_to_prediction} \n\n"
        )
    # Format predictions
    predictions = {
        "version": "1.0",
        "challenge": "ego4d_vq2d_challenge",
        "results": {
            "videos": []
        }
    }
    for v in annotations["videos"]:
        video_predictions = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                auid = a["annotation_uid"]
                if auid in annotation_uid_to_prediction:
                    apred = {
                        
                        "query_sets": annotation_uid_to_prediction[auid], # [0],  # bugs happen...
                        'annotation_uid': auid,
                    }
                else:
                    apred = {
                        "query_sets": {
                            qid: {
                                "bboxes": [],
                                "score": 0.0
                            }
                            for qid in a["query_sets"].keys()
                        },
                        'annotation_uid': auid,
                    }
                clip_predictions["predictions"].append(apred)
            video_predictions["clips"].append(clip_predictions)
        predictions["results"]["videos"].append(video_predictions)
    return predictions


@hydra.main(config_path="vq2d", config_name="config")
def main(cfg: DictConfig) -> None:
    # * note that we're loading the .json.gz file instead of .json file, Load annotations,
    annotations = json.load(open(cfg.data.annot_root))
    annotations_list = convert_annotations_to_list(annotations)
    if cfg.data.debug_mode:
        cfg.data.debug_count = 1
        annotations_list = annotations_list[:cfg.data.debug_count]

    elif cfg.data.subsample:
        annotations_list = annotations_list[::3]
    predicted_rts = predict_vq_test_parallel(annotations_list, cfg)
    # Convert prediction to challenge format
    if cfg.data.debug_mode:
        print(
            f"Check the jsons \n {annotations_list} \n {predicted_rts} \n {cfg} "
        )
    #     [{'metadata': {'video_uid': '17607d8a-806f-4990-b5e7-756b6389c826', 'video_start_sec': 0.0, 'video_end_sec': 480.0, 'clip_fps': 5.0, 'query_set': '2', 'annotation_uid': '09109295-5bc4-4f62-ba44-16e6f6c5301d'}, 
    # 'clip_uid': '634a5fae-5d81-460e-b82b-b5aaa4d77cb3', 'query_frame': 393, 
    # 'visual_crop'
    # * note that for prediction we don't need to format it
    # predictions = format_predictions(annotations,
    #                                  annotations_list,
    #                                  predicted_rts,
    #                                  debug=cfg.data.debug_mode)
    
    predictions={
        'dataset_uids':annotations_list,
        'dataset_predictions':predicted_rts
    }
    
    
    with open(cfg.logging.stats_save_path, "w") as fp:
        json.dump(predictions, fp)

if __name__ == "__main__":
    main()
