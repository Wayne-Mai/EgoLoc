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
    convert_annot_to_bbox,
    get_clip_name_from_clip_uid,
    perform_retrieval,
    SiamPredictor,
)
from vq2d.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack
from vq2d.tracking import Tracker
from pathlib import Path

setup_logger()

import hydra
from omegaconf import DictConfig, OmegaConf
import datetime


SKIP_UIDS = []


def get_images_at_peak(all_bboxes, all_scores, all_imgs, peak_idx, topk=5):
    bboxes = all_bboxes[peak_idx]
    scores = all_scores[peak_idx]
    image = all_imgs[peak_idx]
    # Visualize the top K retrievals from peak image
    bbox_images = []
    for bbox in bboxes[:topk]:
        bbox_images.append(image[bbox.y1:bbox.y2 + 1, bbox.x1:bbox.x2 + 1])
    return bbox_images


def evaluate_vq(annotations, cfg, device_id, use_tqdm=False):

    data_cfg = cfg.data
    dataset_uids = []
    dataset_predictions = []

    device = torch.device(f"cuda:{device_id}")

    # Create detectorvis
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
    # tracker = Tracker(cfg)

    # Visualization
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    # cfg.logging.visualize=False
    if cfg.logging.visualize:
        OmegaConf.save(cfg, os.path.join(cfg.logging.save_dir, "config.yaml"))

    annotations_iter = tqdm.tqdm(annotations) if use_tqdm else annotations
    for idx, annotation in enumerate(annotations_iter):
        start_time = time.time()
        clip_uid = annotation["clip_uid"]
        if clip_uid in SKIP_UIDS:
            continue
        # Load clip from file
        # print(f"so the data root {data_cfg.data_root} and the clip uid {clip_uid}")
        # exit()
        # * actually they're tracking 5 FPS videos
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
        
        
        # * note : code from frost, save the predictions
        ret_bboxes = [[bbox.to_json() for bbox in ret_bbox] for ret_bbox in ret_bboxes]
        prediction = {
            'top_boxes_per_clip': ret_bboxes, # 1081, [bbox_obj_fno_0, bbox_fno_0, .....]
            'top_scores_per_clip':ret_scores, # 1081, [[score, score, .....]]color_00000
            'clip_uid':annotation['clip_uid'],
            'query_frame':annotation['query_frame'],
            'visual_crop':annotation['visual_crop'],
            'visual_crop_fno':annotation['visual_crop']['frame_number'],
            'response_track': sorted(
                annotation['response_track'], key=lambda x: x['frame_number']
            )
        }
        dataset_predictions.append(prediction)
        dataset_uids.append(annotation["dataset_uid"])
        
        
        # start_time = time.time()
        # # Generate a time signal of scores
        # score_signal = []
        # for scores in ret_scores:
        #     if len(scores) == 0:
        #         score_signal.append(0.0)
        #     else:
        #         score_signal.append(np.max(scores).item())
        # # Smooth the signal using median filtering
        # kernel_size = sig_cfg.smoothing_sigma
        # if kernel_size % 2 == 0:
        #     kernel_size += 1
        # score_signal_sm = medfilt(score_signal, kernel_size=kernel_size)
        # # Identify the latest peak in the signal
        # peaks, _ = find_peaks(
        #     score_signal_sm,
        #     distance=sig_cfg.distance,
        #     width=sig_cfg.width,
        #     prominence=sig_cfg.prominence,
        # )
        # peak_signal_time_taken = time.time() - start_time
        
        
        

        
        print("====> Data uid: {} | search window :{:>8d} frames | "
              "clip read time: {:>6.2f} mins | "
              "detection time: {:>6.2f} mins | ".format(
                  annotation["dataset_uid"],
                  annotation["query_frame"],
                  clip_read_time / 60.0,
                  detection_time_taken / 60.0,
                  
              ))


    return (
        dataset_uids,
        dataset_predictions,
    )


def _mp_aux_fn(inputs):
    return evaluate_vq(*inputs)


def evaluate_vq_parallel(annotations, cfg):
    if cfg.data.debug_mode:
        print("Using debug mode...")
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
    print(f"evaluate using {devices}")
    mp_cfgs = [cfg for _ in range(N)]
    mp_devices = [devices[i % len(devices)] for i in range(N)]
    mp_inputs = zip(mp_annotations, mp_cfgs, mp_devices)
    # Perform task
    print(f"Starting {N} threads for processing...")
    # total : N (the expected number of iterations)
    #
    list_of_outputs = list(tqdm.tqdm(pool.imap(_mp_aux_fn, mp_inputs),
                                     total=N))

    # extract predictions
    dataset_uids=[]
    dataset_predictions=[]
    for output in list_of_outputs:
        dataset_uids += output[0]
        dataset_predictions += output[1]
    
    predictions={
        'dataset_uids':dataset_uids,
        'dataset_predictions':dataset_predictions
    }
        
    return predictions


@hydra.main(config_path="vq2d", config_name="config")
def main(cfg: DictConfig) -> None:
    # * note that we're loading the .json.gz file instead of .json file, Load annotations,
    annotations = json.load(open(cfg.data.annot_root))

    if cfg.data.debug_mode:
        annotations = annotations[:cfg.data.debug_count]
    elif cfg.data.subsample:
        annotations = annotations[::3]

    cfg.logging.save_dir = os.path.join(
        cfg.logging.save_dir,
        datetime.datetime.now().strftime("%b-%d_%H-%M"))
    Path(cfg.logging.save_dir).mkdir(parents=True, exist_ok=True)
    predictions = evaluate_vq_parallel(annotations, cfg)
    
    
    print("==========> Saving all the detection results ...")
    json.dump(predictions, open(cfg.logging.stats_save_path, "w"))


if __name__ == "__main__":
    main()
