# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

# from application_util import preprocessing
# from application_util import visualization
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
from my_deep_sort.application_util import preprocessing
from my_deep_sort.application_util import visualization
from my_deep_sort.deep_sort import nn_matching
from my_deep_sort.deep_sort.detection import Detection
from my_deep_sort.deep_sort.tracker import Tracker

from my_deep_sort.utils.data import LingshuiFrameDataset, MOT16TrainFrameDataset, ShenlanFrameDataset
from my_deep_sort.utils.detector import Detector_mmdet, Detector_mmdet_only_instances
from my_deep_sort.utils.encoder import ApparentFeatureExtracker, ApparentFeatureFakeCopier, ApparentFeatureCopier, ApparentFeatureBlocker
from my_deep_sort.utils.evaluator import EvaluatorOfflineMotmetrics

import random
import deep_sort_app


# 观测手段：图像识别检测器
# config_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
# checkpoint_file_detector = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth'
# detector = Detector_mmdet(config_file, checkpoint_file_detector)
detector = None
def setDetector(args):
    global detector
    # detector = Detector_mmdet(args.config_file_detector, args.checkpoint_file_detector)
    detector = Detector_mmdet_only_instances(args.config_file_detector, args.checkpoint_file_detector)

# 卡尔曼滤波更新辅助工具：表观特征抽取器
apExtrackers = {
    'fakecopy': ApparentFeatureFakeCopier(''),  # 直接循环复制 resources/detections/MOT16_POI_test/MOT16-06.npy 中的特征
    'copy':     ApparentFeatureCopier(), # 只能用于 MOT16/train 数据集
    'res18':    ApparentFeatureExtracker('/home/xxy/deep_sort/my_deep_sort/detector_checkpoints/renet18/epoch300.pth'),
    'block':    ApparentFeatureBlocker()
}


def create_detections(img, args):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    img : img 为单张图片的路径，或np.ndarray
    args : parser.parse_args()

    Parameters in args
    ----------
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
        default=0
    apExtractor_type : 'fakecopy', 'copy', 'res18' or 'block'  for now.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    assert args.apExtractor_type in apExtrackers
    apExtracker = apExtrackers[args.apExtractor_type]
    
    detections  = detector.detect(img) #( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
    masks       = detections[1][0] # [100个mask：np.array(..),np.array()，..]
    detections  = detections[0][0] # np.array(100个检测框：[x1,y1,w,h,confidence],[],..)
    detections  = [ (*d,*apExtracker.extract_feature(img, d)) for d in detections] # [ (tlwh,confidence,feature),(),.. ],  img:cv2的image

    detection_list = []
    for d in detections: # row.shape: (138)
        bbox, confidence, feature = d[:4], d[4], d[5:]  # box: tlwh
        if bbox[3] < args.min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature)) # feature.shape:(128)
    return detection_list, masks # [ (tlwh,confidence,feature),(),.. ], [100个mask：np.array(..),np.array()，..]

def get_center(tlwh):
    x,y,w,h = tlwh
    return int(x+w/2), int(y+h/2)

def run(args):
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args)
    tracker = Tracker(metric, args)
    results = []            # [ [frame_idx, track_id, *tlwh],[],.. ]
    trails  = dict()        # { track_id:[[本track最近trail_len帧的轨迹记录点],最近的frame_id] } 轨迹记录点：det框中心 或 mask质心
                            # trails 没有删除失效trail ，待更新
    trail_len = 10          # 每条轨迹记录的最多历史帧数
    track_colors = dict()   # { track_id: (r,g,b) } 用于统一条轨迹和其框的颜色

    # Run tracker.
    # 要推理的图片集合
    if args.dataset == 'lingshui':
        dataset = LingshuiFrameDataset()
    elif args.dataset == 'shenlan1':
        dataset = ShenlanFrameDataset()
    elif args.dataset == 'shenlan':
        dataset = ShenlanFrameDataset('datasets/shenlan/145148-vv-1_full')
    else:   # ['mot16-02', 'mot16-04', 'mot16-05', 'mot16-09', 'mot16-10', 'mot16-11', 'mot16-13']
        dataset = MOT16TrainFrameDataset(args)

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        img = dataset[frame_idx]
        if args.dataset in ['lingshui', 'shenlan1', 'shenlan']:
            detections, masks = create_detections(img, args)
        else:   # ['mot16-02', 'mot16-04', 'mot16-05', 'mot16-09', 'mot16-10', 'mot16-11', 'mot16-13']
            sequence_dir = MOT16TrainFrameDataset.sequence_dir[args.dataset]
            detection_file = MOT16TrainFrameDataset.detection_file[args.dataset]
            seq_info = deep_sort_app.gather_sequence_info(sequence_dir, detection_file)
            detections = deep_sort_app.create_detections( # seq_info["detections"].shape：(10853, 138)
                seq_info["detections"], frame_idx, args.min_detection_height)
            masks = [None]*len(detections) # mot16没有mask

        # detections: [ (tlwh,confidence,feature),(),.. ], masks: [100个mask：np.array(..),np.array()，..]
        # detections = [d for d in detections if d.confidence >= min_confidence]
        tmp_detections, tmp_masks = [], []
        for i,d in enumerate(detections):
            if d.confidence >= args.min_confidence:
                tmp_detections.append(d)
                tmp_masks.append(masks[i])
        detections, masks = tmp_detections, tmp_masks
        # detections: [ (tlwh,confidence,feature),(),.. ], masks: [np.array(..),np.array()，..]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, args, scores)
        detections = [detections[i] for i in indices]

        # Update tracker. 
        tracker.predict()
        det_id_2_track_id, matches, unmatched_tracks, unmatched_detections = tracker.update(detections)  # detections: [ (tlwh,confidence,feature),(),.. ]
            # det_id_2_track_id 第一部分来自于matches，第二部分来自于unmatched_detections
            # det_id_2_track_id 用于产生标记展示图象的labels
        trk_labels_in_det_sequence = [str(det_id_2_track_id[det_id]) for det_id in range(len(detections))]
        for det_idx in range(len(detections)):
            track_idx = det_id_2_track_id[det_idx]
            random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            track_colors.setdefault(track_idx, random_color)
            # if track_idx not in track_colors:
            #     track_colors[track_idx] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        # 每个det都有match的trk，或者新trk，因此每个det都有对应的trk
        trk_colors_in_det_sequence = [track_colors[det_id_2_track_id[det_id]] for det_id in range(len(detections))]

        
        # Update visualization.
        if args.display == True:
            vis.set_image(img.copy())
            if args.draw_detections:
                vis.draw_detections(detections, labels=trk_labels_in_det_sequence, colors=trk_colors_in_det_sequence)    # detections: [ (tlwh,confidence,feature),(),.. ]
            if args.draw_tracks:
                vis.draw_trackers(tracker.tracks, track_colors=track_colors)  
            if args.draw_masks:
                masks = [masks[i] for i in indices]
                vis.draw_detection_masks(masks, labels=trk_labels_in_det_sequence, colors=trk_colors_in_det_sequence) 
                # 应该画trks对应的dets的masks，而不是直接dets的masks
                # 每个det都有match的trk，或者新trk，因此每个det都有对应的trk
                # pass
            if args.draw_trails:
                for detection_idx in det_id_2_track_id: # trails: { track_id:[[本track最近trail_len帧的轨迹记录点],最近的frame_id] } 轨迹记录点：det框中心 或 mask质心
                    track_idx = det_id_2_track_id[detection_idx]
                    if track_idx not in trails:
                        trails[track_idx] = [[detections[detection_idx].get_center()], frame_idx]
                    else:
                        trails[track_idx][0].append(detections[detection_idx].get_center())
                        trails[track_idx][1] = frame_idx
                    trails[track_idx][0] = trails[track_idx][0][-trail_len:]
                vis.draw_trails(trails, frame_idx, colors=trk_colors_in_det_sequence) # 该函数逻辑颜色不匹配，待更新

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    frame_idx = 0 
    # last_idx = dataset.size - 1
    # last_idx = 150
    last_idx = min(args.max_frames - 1,dataset.size - 1) if args.max_frames!=-1 else dataset.size - 1
    output_image_folder = \
        os.path.splitext(args.output_file)[0]
    first_idx, last_idx, image_shape, image_names = \
        frame_idx,last_idx,dataset.image_shape[::-1],dataset.image_names
    if args.display == True:
        visualizer = visualization.Visualization_only_save_image(
            output_image_folder, first_idx, last_idx, image_shape, image_names)
    else:
        visualizer = visualization.NoVisualization(first_idx, last_idx)
    visualizer.run(frame_callback)

    
    # Store results.
    # gt 里边是按照轨迹的 ID 号进行排序的
    results = sorted(results, key=lambda result:result[1])
    os.makedirs(os.path.split(args.output_file)[0], exist_ok=True)  # 保证写文件的目录存在
    f = open(args.output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    if args.display and args.build_video:
        imagename_patern = r"%04d.jpg" if args.dataset in ['lingshui', 'shenlan1'] else \
                           r"%05d.jpg" if args.dataset in ['shenlan'] else \
                           r"%06d.jpg"
        build_video_cmd = f'ffmpeg -r {args.video_fps} -i {os.path.join(output_image_folder,imagename_patern)} {output_image_folder+".mp4"} -y'
        os.system(build_video_cmd)
    if args.dataset not in ['lingshui', 'shenlan1', 'shenlan']:
        sequence_dir = MOT16TrainFrameDataset.sequence_dir[args.dataset]
        ground_truth_file = os.path.join(sequence_dir,'gt/gt.txt')
        track_result_file = args.output_file
        EvaluatorOfflineMotmetrics(ground_truth_file, track_result_file, 'critical', 'mot16')


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
        "--dataset", help="Select one of MOT16 sets or lingshui-set.",
        default="lingshui",
        choices=['lingshui', 'shenlan1', 'shenlan', 'mot16-02', 'mot16-04', 'mot16-05', 'mot16-09',
                'mot16-10', 'mot16-11', 'mot16-13'])
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--device", help="模型和数据的存放运行设备。",
        default='cuda:1', type=str,
        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument(
        "--config_file_detector", help="检测器定义文件。",
        default='/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py',
        type=str)
    parser.add_argument(
        "--checkpoint_file_detector", help="检测器权重文件。",
        default='/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth',
        type=str)
    parser.add_argument(
        "--max_frames", help="从数据集识别并跟踪的最大帧数，默认30，-1代表数据集中所有帧。",
        default='30', type=int)
    parser.add_argument(
        "--min_height", help="A minimum detection bounding box height. "
        "Detections that are smaller than this value are disregarded.",
        default=0, type=int)
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
        "detection overlap. \n ROIs that overlap more than this values are suppressed.", 
        default=1.0, type=float)
    parser.add_argument(
        "--max_age",  help="Maximum number of missed misses before a track is deleted.",
        default=30, type=int)
    parser.add_argument(
        "--n_init",  help="Number of frames that a track remains in initialization phase. \n" + 
        "Number of consecutive detections before the track is confirmed. The track state " +
        "is set to `Deleted` if a miss occurs within the first `n_init` frames.",
        default=3, type=int)
    parser.add_argument(
        "--apExtractor_type",  help="'fakecopy', 'copy', 'res18' or 'block' for now.", 
        default='res18', type=str, choices=['fakecopy', 'copy', 'res18', 'block'])
    parser.add_argument(
        "--kalmanFilter_type",  help="'raw', 'xy', 'qr' or 'xyqr' for now.", 
        default='raw', type=str, choices=['raw', 'xy', 'qr', 'xyqr'])
    parser.add_argument(
        "--Q_times",  help="卡尔曼滤波器的超参数，用于调整 超参数Q（过程噪声协方差） 的大小。", 
        default=1.0, type=float)
    parser.add_argument(
        "--R_times",  help="卡尔曼滤波器的超参数，用于调整 超参数R（观测噪声协方差） 的大小。", 
        default=1.0, type=float)
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
    parser.add_argument(
        "--video_fps", help="Set the fps of the output .mp4 file.",
        default=30.0, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setDetector(args)
    run(args)
