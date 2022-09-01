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

from my_deep_sort.utils.data import Dataset
from my_deep_sort.utils.detector import Detector
from my_deep_sort.utils.encoder import Encoder


# 观测手段：图像识别检测器
config_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
checkpoint_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth'
detector = Detector(config_file, checkpoint_file)

# 卡尔曼滤波更新辅助工具：表观特征抽取器
encoder = Encoder('', '') # 还没实现，只写了接口extract_feature，因此其返回一个128维随机特征


def create_detections(img, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    img : str
        The frame image file.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    
    detections = detector.detect(img) #( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
    detections = detections[0][0] # np.array(100个检测框：[x1,y1,w,h,confidence],[],..)
    detections = [ (*d,*encoder.extract_feature(img, d)) for d in detections] # [ (tlwh,confidence,feature),(),.. ]

    detection_list = []
    for d in detections: # row.shape: (138)
        bbox, confidence, feature = d[:4], d[4], d[5:]  # box: tlwh
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature)) # feature.shape:(128)
    return detection_list # [ (tlwh,confidence,feature),(),.. ]


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []    # [ [frame_idx, track_id, *tlwh],[],.. ]

    # Run tracker.
    # 要推理的图片集合
    dataset = Dataset()
    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        img = dataset[frame_idx]
        detections = create_detections(
            img, min_detection_height)
        # detections: [ (tlwh,confidence,feature),(),.. ]
        detections = [d for d in detections if d.confidence >= min_confidence]
        # detections: [ (tlwh,confidence,feature),(),.. ]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)  # detections: [ (tlwh,confidence,feature),(),.. ]

        # Update visualization.
        if display:
            image = cv2.imread(
                img, cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    frame_idx = 0 
    # last_idx = dataset.size - 1
    last_idx = 10
    visualizer = visualization.Visualization_only_save_image(
        'output/3fps300s',frame_idx,last_idx,dataset.image_shape[::-1],dataset.image_names)
    visualizer.run(frame_callback)

    
    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


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
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
