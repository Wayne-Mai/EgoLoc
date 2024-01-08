import numpy as np
import torch
from vq3d.depth_utils import get_object_depth



def weighted2d(all_res, k):
    resk = []
    weight_det = []
    weight_clip = []
    weight_clear = []
    for idx, res in enumerate(all_res):
        resk.append(res[k])
        weight_det.append(res['detection_score'])
        weight_clip.append(res['clip_score'])
        weight_clear.append(res['clear_score'])
    resk, weight_det, weight_clip = np.array(resk), np.array(
        weight_det), np.array(weight_clip)
    return np.average(resk, weights=weight_det, axis=0).tolist()


def nms(args,all_res,k):
    # args.nms_threshold=0.3 spatial l2 distance in meters, or 10000
    # args.nms_fusion=True: merge the observation if min>0.5*max clip score
    # args.nms_fusion_threshold: <= 0 means no fusion, always return 
    # top 1>0, means we use weighted average
    
    
    resk=[]
    weight_det=[]
    nms_threshold=args.nms_threshold
    nms_fusion_threshold=args.nms_fusion_threshold
    for idx,res in enumerate(all_res):
        resk.append(res[k])
        weight_det.append(res['detection_score'])
        
    if len(resk)==1:
        return resk[0]
    
    resk, weight_det= np.array(resk), np.array(
        weight_det)
    
    order=weight_det.argsort()[::-1]
    proposals=[]
    proposal_scores=[]
    while order.size>0:
        
        # keep_inds
        # begin fusion step
        # others are starting from 1 to n
        dist=np.linalg.norm(resk[order[1:]]- resk[order[0]], axis=1)
        close_order_inds=np.where(dist<=nms_threshold)[0]
        far_order_inds=np.where(dist>nms_threshold)[0]
        
        close_points_inds=order[close_order_inds+1]
        far_points_inds=order[far_order_inds+1]
        
        fused_point,fused_score=[resk[order[0]]],[weight_det[order[0]]]
        if nms_fusion_threshold>0:
            for ind in close_points_inds:
                if weight_det[ind]>nms_fusion_threshold*weight_det[order[0]]:
                    fused_point.append(resk[ind])
                    fused_score.append(weight_det[ind])
            if k=='mp_pred_3d_vec_world' and len(proposals)==0:
                print(f"Merging {len(fused_point)} observations")
            fused_point,fused_score=np.array(fused_point),np.array(fused_score)
            fused_point=np.average(fused_point,weights=fused_score,axis=0)
        else:
            fused_point=fused_point[0]
        
        proposals.append(fused_point)
        proposal_scores.append(fused_score[0])
        order=far_points_inds
    if k=='mp_pred_3d_vec_world':
        print(f"Aggregate to ",proposals[0], "\n")
    
    selected_idx=0
    if 'nms_post' in args.experiment:
        total_dist=100000
        
        for idx in range(len(proposals)):
            t_dist=np.sum(np.linalg.norm(resk-proposals[idx], axis=1))
            if t_dist<total_dist and proposal_scores[idx]>nms_fusion_threshold*proposal_scores[0]:
                selected_idx=idx
                total_dist=t_dist
            
    return proposals[selected_idx].tolist()




def predict_for_frame_j(args, frame_indices_valid, local_frame_indices, frames,
                        T, pose_Q, paras_value, depth_dir, center_vectors, j):
    qset = {}

    f, cx, cy, k1, k2 = paras_value
    W, H = cx * 2.0, cy * 2.0

    # get the last frame of the RT
    # j = np.argmax(frame_indices_valid)
    frame_index_valid = frame_indices_valid[
        j]  # select i-th pose from rt with poses
    local_frame_index = local_frame_indices[j]  # select i-th bbox from rt

    # * note : all the '_pose' here are actually extrinsics


    # get RT frame pose
    pose = T[frame_index_valid]

    # cpt_valid_queries += 1

    d, x1, x2, y1, y2 = get_object_depth(depth_dir,
                                         frame_index_valid,
                                         frames[local_frame_index],
                                         H,
                                         W,
                                         use_gt=False)

    if d.size == 0:
        print(f"Waning : no depth for pred frame j:{j}")
        return qset
    d = np.median(d)
    
    # experiment note : only for ablation study
    # d= np.random.uniform(0.1, 10)
    
    tx = (x1 + x2) / 2.0
    ty = (y1 + y2) / 2.0

    # vec in current frame, that is, vec in :
    # z = d/4
    z=d
    x = z * (tx - cx - 0.5) / f
    y = z * (ty - cy - 0.5) / f
    vec = np.ones(4)
    vec[0] = x
    vec[1] = y
    vec[2] = z
    

    # object center in world coord system
    # pred_t = np.matmul( pose, vec)
    pred_t = np.matmul(np.linalg.inv(pose), vec)
    pred_t = pred_t / pred_t[3]
    
    
    
    # save output for metric compute
    # * note : our answer 3
    # extrinsics (colmap, baseline output) is world to camera, how to transform a point from world system to camera system, extrinsics, (uv1)*z=KTPw
    # camera's pose are usually cam2world, how to transform a point in camera system to world system, the location of camera,  C(k-1uv)=Pw
    qset['mp_pred_3d_vec_world'] = pred_t[:3].tolist()
    qset['pred_3d_vec_world'] = [-pred_t[1], pred_t[0], pred_t[2]]
    qset['rt_frame_pose'] = pose.tolist()
    qset['selected_rt_frame'] = frame_index_valid
    qset['selected_rt_bbox'] = [x1, x2, y1, y2]
    qset['clear_score']=1.0
    
    if 'detection_score' in frames[local_frame_index].keys():
        qset['detection_score'] = frames[local_frame_index]['detection_score']
        qset['clip_score'] = frames[local_frame_index]['clip_score']
    else:
        qset['detection_score'] = 1.0
        qset['clip_score'] = 1.0

    # object center in Query frame coord system
    # note that pred_3d_vec_query_frame is depend on the pred_t
    # * note : ans 2
    if pose_Q is not None:
        vec = np.matmul(pose_Q, pred_t)
        vec = vec / vec[3]
        vec = vec[:3]
        qset['mp_pred_3d_vec_query_frame'] = vec.tolist(
        )  # same as pred_3d_vec
        qset['pred_3d_vec_query_frame'] = [-vec[1], vec[0], vec[2]]
        rotated_predict=np.matmul(
            pose_Q, np.array([-pred_t[1], pred_t[0], pred_t[2],
                              1.0]))
        rotated_predict=rotated_predict/rotated_predict[3]
        qset['rotated_pred_3d_vec_query_frame'] = rotated_predict[:3].tolist()

        if len(center_vectors) == 2:
            for w, center_vec in enumerate(center_vectors):
                center_vec = np.append(center_vec, 1.0)

                gt_3d_vec = np.matmul(pose_Q, center_vec)
                gt_3d_vec = gt_3d_vec[:3] / gt_3d_vec[3]

                qset[f'mp_gt_3d_vec_{w}'] = gt_3d_vec.tolist()
                qset[f'gt_3d_vec_{w}'] = [
                    -gt_3d_vec[1], gt_3d_vec[0], gt_3d_vec[2]
                ]
            
    else:
        vec = None
        qset['pred_3d_vec_query_frame'] = None
        qset['rotated_pred_3d_vec_query_frame'] = None
        qset['mp_pred_3d_vec_query_frame'] = None

    return qset



def multi_frames_prediction(
    args,
    all_res,
):
    args_experiment = args.experiment
    res = {}
    if len(all_res) == 0:
        return res

    res['rt_frame_pose'] = all_res[0]['rt_frame_pose']
    res['selected_rt_frame'] = all_res[0]['selected_rt_frame']
    res['selected_rt_bbox'] = all_res[0]['selected_rt_bbox']

    if 'mf' in args_experiment: # multi frame aggregation option
        to_median_key = [
            'mp_pred_3d_vec_world',
            'pred_3d_vec_world',
            'mp_pred_3d_vec_query_frame',
            'pred_3d_vec_query_frame',
            'rotated_pred_3d_vec_query_frame',
            'mp_gt_3d_vec_1',
            # annotations are given by two annotators
            'gt_3d_vec_1',  # annot1 with pose_Q
            'mp_gt_3d_vec_2',
            'gt_3d_vec_2',  # annot2 with pose_Q
        ]

        for k in to_median_key:
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            value_arr = np.array(
                [x[k] for x in all_res if k in x and x[k] is not None])
            # if k == 'mp_pred_3d_vec_world':
            #     print(value_arr)
            if value_arr.shape[0] != 0:
                if 'median' in args_experiment:
                    res[k] = np.median(value_arr, axis=0).tolist()
                elif 'mean' in args_experiment:
                    res[k] = np.mean(value_arr, axis=0).tolist()
                elif 'weighted2d' in args_experiment:
                    res[k] = weighted2d(all_res, k)
                elif 'nms' in args_experiment:
                    res[k] = nms(args,all_res, k)

        return res