# this block is for detection
"""
write codes to get the different detection result 
output format should look like:

predictions 6
	 predicted_response_track 4504
	 ground_truth_response_track 4504
	 visual_crop 4504
	 dataset_uids 4504
	 accessed_frames 4504
	 total_frames 4504
    -> : so actually accessed_frames and total_frames looks the same 
metrics 7

input format look like: (result file)

dataset_uids: ['val_0000']
dataset_predictions:
    top_boxes_per_clip: [[{fno,x1,x2,y1,y2}...]  ]
    top_scores_per_clip: [[0.1,...]   ]
    clip_uid:
    query_frame:
    visual_crop:
    visual_crop_fno:
    response_track: (gt)
    
input file 2: (val_annot_vq3d):

object_title:
dataset_uid:
metadata:
response_track:
query_set:
query_frame:

"""
import os, json
from scipy.signal import find_peaks, medfilt
import numpy as np
from munch import DefaultMunch
import matplotlib.pyplot as plt
from vq2d.structures import ResponseTrack
import datetime
from pathlib import Path
import pims
import skimage.io
import torch
from einops import rearrange, asnumpy
import cv2
from tqdm import tqdm
import clip
import PIL

from vq2d.baselines import (
    create_similarity_network,
    convert_annot_to_bbox,
    get_clip_name_from_clip_uid,
    perform_retrieval,
    SiamPredictor,
    extract_window_with_context
)

signals = {
    'smoothing_sigma': 5, 
    'distance': 25,
    'width': 3,
    'prominence': 0.2,
    'wlen': 50
}

cfgs = {
    'mode': 'all_detpeak',  # 'peak_only',peak_clip, 
    'mode_config': {
        'window_threshold': 0.5
    },
    'logging': {
        'save_dir': 'data/new_logs',  # 'visual_queries_logs',
        'save_file': 'vq2d_val_vq3d.json'
    },
    'vis': False,
    'data_root': 'data/clips',
    'INPUT': {
        'REFERENCE_CONTEXT_PAD': 16,
        'REFERENCE_SIZE': 256,
    }
}

# TODO: use a sigma to select the neighbor as detection response track
# multi view

sig_cfgs = DefaultMunch.fromDict(signals)
cfgs = DefaultMunch.fromDict(cfgs)
detection_scores = json.load(
    open('visual_queries_logs/baseline_vq3d_detection_v105.json'))
metadata = json.load(open('data/val_annot_vq3d.json'))

database=json.load(
    open('data/kw60746_vq2d_results/vq2d_val_vq3d_greedy_peak_clip.json'))

def get_images_at_peak(all_bboxes, all_scores, all_imgs, peak_idx, topk=5):
    bboxes = all_bboxes[peak_idx]
    scores = all_scores[peak_idx]
    image = all_imgs[peak_idx]
    # Visualize the top K retrievals from peak image
    bbox_images = []
    for bbox in bboxes[:topk]:
        if isinstance(bbox, dict):
            bbox_images.append(image[bbox['y1']:bbox['y2'] + 1,
                                     bbox['x1']:bbox['x2'] + 1])
        else:
            bbox_images.append(image[bbox.y1:bbox.y2 + 1, bbox.x1:bbox.x2 + 1])
    return bbox_images


def format_vq2d_detection(responses):
    res = {
        'predictions': {
            'predicted_response_track': [],
            'ground_truth_response_track': [],
            'visual_crop': [],
            'dataset_uids': [],
            'accessed_frames': [],
            'total_frames': []
        },
        'metrics': {}
    }

    for response in responses:
        for k in res['predictions'].keys():
            if k == 'ground_truth_response_track':
                response[k] = response[k].to_json()
            res['predictions'][k].append(response[k])
    return res


def vq2d_detection(detection_scores, metadata, sig_cfg, cfgs):
    responses = []
    gt_response_track = []
    # cfgs.logging.save_dir = os.path.join(
    #     cfgs.logging.save_dir,
    #     datetime.datetime.now().strftime("%b-%d_%H-%M"))
    Path(cfgs.logging.save_dir).mkdir(parents=True, exist_ok=True)

    cfgs.logging.save_file = cfgs.logging.save_file[:
                                                    -5] + '_' + cfgs.mode + cfgs.logging.save_file[
                                                        -5:]
    if 'peak_clip' in cfgs.mode:
        device = 3 if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load('ViT-B/32', device)
        print(f"Using {device} for our clip peak selection ")

    temp_names=cfgs.logging.save_file.split('.')
    cfgs.logging.save_file=temp_names[0]+str(cfgs.mode_config['window_threshold'])+'.'+temp_names[1]
            
    for idx, dataset_uid in tqdm(enumerate(detection_scores['dataset_uids']),
                                 total=len(detection_scores['dataset_uids'])):
        predictions = detection_scores['dataset_predictions'][idx]
        ret_bboxes = predictions['top_boxes_per_clip']
        ret_scores = predictions['top_scores_per_clip']
        visual_crop = predictions['visual_crop']
        clip_uid = predictions['clip_uid']
        assert metadata[idx]['dataset_uid'] == detection_scores[
            'dataset_uids'][idx]
        query_frame = metadata[idx]['query_frame']
        # select the top 1 bbox and use a threshold to get response track
        score_signal = []

        gt_response_track.append(
            ResponseTrack([
                convert_annot_to_bbox(rf)
                for rf in predictions["response_track"]
            ]))

        for scores in ret_scores:
            if len(scores) == 0:
                score_signal.append(0.0)
            else:
                score_signal.append(np.max(scores).item())
        # Smooth the signal using median filtering
        kernel_size = sig_cfg.smoothing_sigma
        if kernel_size % 2 == 0:
            kernel_size += 1
        # start from last query frame
        score_signal_sm = medfilt(score_signal, kernel_size=kernel_size)

        # Identify the latest peak in the signal
        peaks, _ = find_peaks(
            score_signal_sm,
            distance=sig_cfg.distance,
            width=sig_cfg.width,
            prominence=sig_cfg.prominence,
        )
        # peaks: [fno_1,...fno_k]
        

        ####################### format the 2d detection result ########################
        accessed_frames = set()
        for bboxes in ret_bboxes:
            accessed_frames.add(bboxes[0]['fno'])

        response = {
            # 'predicted_response_track': [],
            'ground_truth_response_track': gt_response_track[-1],
            'visual_crop': visual_crop,
            'dataset_uids': dataset_uid,
            'accessed_frames': len(accessed_frames),
            'total_frames': query_frame
        }
       
        
        # peaks=[]
        if len(peaks)==0:
            print(f"Warning : No peak in idx {dataset_uid} we use all")
            peaks=list(range(len(ret_bboxes)))
            
        if cfgs.vis or 'peak_clip' in cfgs.mode:
            ####################### Prepare frames for visualization ########################
            clip_path = os.path.join(cfgs.data_root,
                                     get_clip_name_from_clip_uid(clip_uid))
            video_reader = pims.Video(clip_path)
            clip_frames = video_reader[
                0:max(query_frame, visual_crop['frame_number']) + 1]

            ###################### Visualize visual crop ########################
            vc_fno = visual_crop["frame_number"]
            owidth, oheight = visual_crop["original_width"], visual_crop[
                "original_height"]

            # Load visual crop frame
            reference = clip_frames[vc_fno]  # RGB format
            ## Resize visual crop if stored aspect ratio was incorrect
            if (reference.shape[0] != oheight) or (reference.shape[1] !=
                                                   owidth):
                reference = cv2.resize(reference, (owidth, oheight))
            reference = torch.as_tensor(
                rearrange(reference, "h w c -> () c h w"))
            reference = reference.float()
            ref_bbox = (
                visual_crop["x"],
                visual_crop["y"],
                visual_crop["x"] + visual_crop["width"],
                visual_crop["y"] + visual_crop["height"],
            )
            reference = extract_window_with_context(
                reference,
                ref_bbox,
                cfgs.INPUT.REFERENCE_CONTEXT_PAD,
                size=cfgs.INPUT.REFERENCE_SIZE,
                pad_value=125,
            )

            if cfgs.vis:
                visual_crop_im = rearrange(asnumpy(reference.byte()),
                                           "() c h w -> h w c")
                # Visualize crop
                save_path = os.path.join(cfgs.logging.save_dir,
                                         f"example_{idx:05d}_visual_crop.png")
                skimage.io.imsave(save_path, visual_crop_im)

            ####################### USE CLIP to SELECT PEAKS ########################

            ###################### Visualize retrievals ########################
            # Define search window
            search_window = list(range(0, query_frame))
            ret_imgs = [clip_frames[j] for j in search_window]
            top_1_proposals = []
            for peak_idx in peaks:
                peak_images = get_images_at_peak(ret_bboxes,
                                                 ret_scores,
                                                 ret_imgs,
                                                 peak_idx,
                                                 topk=5)
                top_1_proposals.append(
                    clip_preprocess(
                        PIL.Image.fromarray(peak_images[0])).to(device))
                for image_idx, image in enumerate(peak_images):
                    if cfgs.vis:
                        save_path = os.path.join(
                            cfgs.logging.save_dir,
                            f"example_{idx:05d}_peak_{peak_idx:05d}_rank_{image_idx:03d}.png",
                        )
                        skimage.io.imsave(save_path, image)
            if 'peak_clip' in cfgs.mode:
                top_1_proposals = torch.stack(top_1_proposals)
                # print(top_1_proposals.shape)

        if 'peak_only' in cfgs.mode:
            temp_bboxes = []
            for peak_idx in peaks:
                temp_bboxes.append(ret_bboxes[peak_idx][0])
        elif 'peak_clip' in cfgs.mode:
            ref_image= clip_preprocess(
                        PIL.Image.fromarray(clip_frames[vc_fno])).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features=clip_model.encode_image(top_1_proposals)
                ref_feature=clip_model.encode_image(ref_image)
                
            image_features/=image_features.norm(dim=-1,keepdim=True)
            ref_feature/=ref_feature.norm(dim=-1,keepdim=True)
            similarity=(100.0*ref_feature@image_features.T).softmax(dim=-1).float() 
            # tensor([0.3665, 0.2223, 0.0986, 0.0678, 0.0528] tensor([ 1, 10,  8, 16,  3]
            temp_bboxes = []
            for peak_idx,clip_score in zip(peaks, similarity[0]):
                temp_bboxes.append(ret_bboxes[peak_idx][0])
                temp_bboxes[-1]['clip_score']=clip_score.item()
                temp_bboxes[-1]['detection_score']=ret_scores[peak_idx][0]
            
            del image_features
            del ref_feature
            torch.cuda.empty_cache() 
        
        elif 'detwindow' in cfgs.mode:
            
            db_bboxes=database['predictions']['predicted_response_track'][idx]['bboxes']
            clip_score=[bbox['clip_score'] for bbox in db_bboxes]
            det_score=[bbox['detection_score'] for bbox in db_bboxes]
            last_peak=peaks[-1]
            temp_bboxes=[]
            
            # forward
            current=last_peak
            # and clip_score[current]>=clip_score[last_peak]*cfgs.mode_config.window_threshold
            while current>=0 and det_score[current]>=det_score[last_peak]*cfgs.mode_config.window_threshold:
                temp_bboxes.insert(0,ret_bboxes[current][0])
                current-=1
                
            # backward
            current=last_peak+1
            # and clip_score[current]>=clip_score[last_peak]*cfgs.mode_config.window_threshold
            while current<len(clip_score) and det_score[current]>=det_score[last_peak]*cfgs.mode_config.window_threshold:
                temp_bboxes.append(ret_bboxes[current][0])
                current+=1
        elif 'all_detpeak' in cfgs.mode:
            
            db_bboxes=database['predictions']['predicted_response_track'][idx]['bboxes']
            clip_score=[bbox['clip_score'] for bbox in db_bboxes]
            det_score=[bbox['detection_score'] for bbox in db_bboxes]
            last_peak=peaks[-1]
            temp_bboxes=[]
            
            for last_peak in peaks:
                # forward
                current=last_peak
                # and clip_score[current]>=clip_score[last_peak]*cfgs.mode_config.window_threshold
                while current>=0 and det_score[current]>=det_score[last_peak]*cfgs.mode_config.window_threshold:
                    temp_bboxes.insert(0,ret_bboxes[current][0])
                    current-=1
                    
                # backward
                current=last_peak+1
                # and clip_score[current]>=clip_score[last_peak]*cfgs.mode_config.window_threshold
                while current<len(clip_score) and det_score[current]>=det_score[last_peak]*cfgs.mode_config.window_threshold:
                    temp_bboxes.append(ret_bboxes[current][0])
                    current+=1
                    
        elif 'clipwindow' in cfgs.mode:
            db_bboxes=database['predictions']['predicted_response_track'][idx]['bboxes']
            clip_score=[bbox['clip_score'] for bbox in db_bboxes]
            det_score=[bbox['detection_score'] for bbox in db_bboxes]
            last_peak_idx=np.argmax([clip_score[p] for p in peaks])
            last_peak=peaks[last_peak_idx]
            temp_bboxes=[]
            
            # forward
            current=last_peak
            while current>=0 and clip_score[current]>=clip_score[last_peak]*cfgs.mode_config.window_threshold and det_score[current]>=det_score[last_peak]*cfgs.mode_config.window_threshold:
                temp_bboxes.insert(0,ret_bboxes[current][0])
                current-=1
                
            # backward
            current=last_peak+1
            while current<len(clip_score) and clip_score[current]>=clip_score[last_peak]*cfgs.mode_config.window_threshold and det_score[current]>=det_score[last_peak]*cfgs.mode_config.window_threshold:
                temp_bboxes.append(ret_bboxes[current][0])
                current+=1    
            
        response['predicted_response_track'] = {'bboxes': temp_bboxes}
        responses.append(response)

        if cfgs.vis:

            ####################### Visualize the peaks ########################
            plt.plot(score_signal_sm, color="blue", label="Similarity scores")
            # Plot peak in signal
            plt.plot(peaks, score_signal_sm[peaks], "rx", label="Peaks")
            # Get GT response window
            gt_rt_start, gt_rt_end = gt_response_track[-1].temporal_extent
            rt_signal = np.zeros((query_frame, ))
            rt_signal[gt_rt_start:gt_rt_end + 1] = 1
            plt.plot(rt_signal, color="green", label="GT Response track")
            plt.legend()
            plt.savefig(save_path, dpi=500)
            plt.close()
            # Print the result
            print("\nTop predictions:\n")
            values, indices = similarity[0].topk(5)
            for value, index in zip(values, indices):
                print(f"{index:>4}: {100 * value.item():.2f}%")
        
        # break

    all_res = format_vq2d_detection(responses)
    json.dump(
        all_res,
        open(os.path.join(cfgs.logging.save_dir, cfgs.logging.save_file), 'w'))

    return all_res


_ = vq2d_detection(detection_scores, metadata, sig_cfgs, cfgs)
