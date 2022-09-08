# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from my_deep_sort.utils.data import LingshuiFrameDataset
from my_deep_sort.utils.detector import Detector
from my_deep_sort.utils.encoder import ApparentFeatureExtracker, ApparentFeatureCopier

import random


# 观测手段：图像识别检测器
config_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
checkpoint_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth'
detector = Detector(config_file, checkpoint_file)

# 卡尔曼滤波更新辅助工具：表观特征抽取器
# apExtracker = ApparentFeatureExtracker('') # 还没实现，只写了接口extract_feature，因此其返回一个128维随机特征
apExtracker = ApparentFeatureCopier('') # 直接循环复制 resources/detections/MOT16_POI_test/MOT16-06.npy 中的特征


def create_detections(img, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    img : img 为单张图片的路径，或np.ndarray
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    
    detections  = detector.detect(img) #( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
    masks       = detections[1][0] # [100个mask：np.array(..),np.array()，..]
    detections  = detections[0][0] # np.array(100个检测框：[x1,y1,w,h,confidence],[],..)
    detections  = [ (*d,*apExtracker.extract_feature(img, d)) for d in detections] # [ (tlwh,confidence,feature),(),.. ]

    detection_list = []
    for d in detections: # row.shape: (138)
        bbox, confidence, feature = d[:4], d[4], d[5:]  # box: tlwh
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature)) # feature.shape:(128)
    return detection_list, masks # [ (tlwh,confidence,feature),(),.. ], [100个mask：np.array(..),np.array()，..]

def get_center(tlwh):
    x,y,w,h = tlwh
    return int(x+w/2), int(y+h/2)

def run(output_file, build_video, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, draw_masks, draw_detections, draw_tracks, draw_trails):

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []            # [ [frame_idx, track_id, *tlwh],[],.. ]
    trails  = dict()        # { track_id:[[本track最近trail_len帧框的中心点],最近的frame_id] }
    trail_len = 10          # 每条轨迹记录的最多历史帧数
    track_colors = dict()   # { track_id: (r,g,b) } 用于统一条轨迹和其框的颜色

    # Run tracker.
    # 要推理的图片集合
    dataset = LingshuiFrameDataset()
    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        img = dataset[frame_idx]
        detections, masks = create_detections(
            img, min_detection_height)
        # detections: [ (tlwh,confidence,feature),(),.. ], masks: [100个mask：np.array(..),np.array()，..]
        # detections = [d for d in detections if d.confidence >= min_confidence]
        tmp_detections, tmp_masks = [], []
        for i,d in enumerate(detections):
            if d.confidence >= min_confidence:
                tmp_detections.append(d)
                tmp_masks.append(masks[i])
        detections, masks = tmp_detections, tmp_masks
        # detections: [ (tlwh,confidence,feature),(),.. ], masks: [np.array(..),np.array()，..]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        masks = [masks[i] for i in indices]

        # Update tracker. 
        tracker.predict()
        det_id_2_track_id = tracker.update(detections)  # detections: [ (tlwh,confidence,feature),(),.. ]
            # det_id_2_track_id 第一部分来自于matches，第二部分来自于unmatched_detections
            # det_id_2_track_id 用于产生标记展示图象的labels
        labels = [str(det_id_2_track_id[det_id]) for det_id in range(len(detections))]
        for det_idx in range(len(detections)):
            track_idx = det_id_2_track_id[det_idx]
            if track_idx not in track_colors:
                track_colors[track_idx] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        det_colors = [track_colors[det_id_2_track_id[det_id]] for det_id in range(len(detections))]

        for detection_idx in det_id_2_track_id: # trails: { track_id:[[本track最近trail_len帧框的中心点],最近的frame_id] }
            track_idx = det_id_2_track_id[detection_idx]
            if track_idx not in trails:
                trails[track_idx] = [[detections[detection_idx].get_center()], frame_idx]
            else:
                trails[track_idx][0].append(detections[detection_idx].get_center())
                trails[track_idx][1] = frame_idx
            trails[track_idx][0] = trails[track_idx][0][-trail_len:]

        # Update visualization.
        vis.set_image(img.copy())
        if draw_detections:
            vis.draw_detections(detections, labels, colors=det_colors)    # detections: [ (tlwh,confidence,feature),(),.. ]
        if draw_tracks:
            vis.draw_trackers(tracker.tracks, track_colors=track_colors)  
        if draw_masks:
            vis.draw_detection_masks(masks, colors=det_colors, labels=labels)
        if draw_trails:
            vis.draw_trails(trails, frame_idx, colors=det_colors)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    frame_idx = 0 
    # last_idx = dataset.size - 1
    last_idx = 150
    output_image_folder = \
        os.path.splitext(output_file)[0]
    first_idx, last_idx, image_shape, image_names = \
        frame_idx,last_idx,dataset.image_shape[::-1],dataset.image_names
    visualizer = visualization.Visualization_only_save_image(
        output_image_folder, first_idx, last_idx, image_shape, image_names)
    visualizer.run(frame_callback)

    
    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    if build_video:
        build_video_cmd = f'ffmpeg -r 30 -i {os.path.join(output_image_folder,r"%04d.jpg")} {output_image_folder+".mp4"} -y'
        os.system(build_video_cmd)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--draw_masks", help="Draw detections' masks.",
        default=True, type=bool_string)
    parser.add_argument(
        "--draw_detections", help="Draw detections' bboxes.",
        default=False, type=bool_string)
    parser.add_argument(
        "--draw_tracks", help="Draw track' bboxes.",
        default=False, type=bool_string)
    parser.add_argument(
        "--draw_trails", help="Draw trails of same tracks.",
        default=False, type=bool_string)
    parser.add_argument(
        "--build_video", help="Build a .mp4 file with outputed images.",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_file, args.build_video,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, 
        args.draw_masks, args.draw_detections, args.draw_tracks, args.draw_trails)
