from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from utils.data import Dataset
from utils.detector import Detector
from utils.encoder import Encoder
import numpy as np
from application_util import preprocessing
from deep_sort.detection import Detection

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
    
    # 观测手段：图像识别检测器
    config_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    checkpoint_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth'
    detector = Detector(config_file, checkpoint_file)

    detections = detector.detect(img) #( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
    detections = detections[0][0] # np.array(100个检测框：[x1,y1,w,h,confidence],[],..)
    detections = [ (*d,encoder.extract_feature(img, d)) for d in detections] # [ (tlwh,confidence,feature),(),.. ]

    detection_list = []
    for d in detections: # row.shape: (138)
        bbox, confidence, feature = d[:4], d[4], d[5:]  # box: tlwh
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature)) # feature.shape:(128)
    return detection_list # [ (tlwh,confidence,feature),(),.. ]

if __name__ == "__main__":    # 就是def run():
    # 要推理的图片集合
    dataset = Dataset()
    frame_idx = 0      # 这两个均在visualizer中定义，精简后可视为在deep_sort_app.py中定义
    last_idx = dataset.size - 1

    
    # 卡尔曼滤波更新辅助工具：表观特征抽取器
    encoder = Encoder('', '') # 还没实现，只写了接口extract_feature，因此其返回一个128维随机特征

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, 70)
    tracker = Tracker(metric)
    results = []             # [ [frame_idx, track_id, *tlwh],[],.. ]
    
    while frame_idx <= last_idx:  # 精简结合自visualizer.run(frame_callback) 和 def frame_callback(.., frame_idx)
        img = dataset[frame_idx]
        # detections = …       # [ (tlwh,confidence,feature),(),.. ]
        detections = create_detections( img, 0)
        # nms to detections
        boxes = np.array([d.tlwh for d in detections]) # tlwh
        scores = np.array([d.confidence for d in detections]) # confidence
        indices = preprocessing.non_max_suppression(
            boxes, 1.0, scores)
        detections = [detections[i] for i in indices]
        
        tracker.predict()
        tracker.update(detections)  # detections: [ (tlwh,confidence,feature),(),.. ]
        # 将tracker.tracks中 ( track.is_confirmed() and track.time_since_update <= 1 ) 的track加入 results 
                                # track: [frame_idx, track_id, *tlwh]
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        frame_idx += 1

    output_file = 'tmp_output.txt'
    with open(output_file, 'w') as f:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)