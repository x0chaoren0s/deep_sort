# https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/1_exist_data_model.md

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import mmcv
import numpy as np
import cv2

class Detector_base:
    '''
    所有Detector继承于此基类，并需要重载 self.model 和 self.detect(self, img)
    '''
    def __init__(self) -> None:
        self.model = None
    
    def detect(self, img):
        ''' 
        暂时只支持 img 为单张图片的路径，或np.ndarray。\n
        返回一个tuple：( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
        '''
        raise ValueError('not implemented.')

class Detector_mmdet(Detector_base):
    '''
    适配mmlab
    使用 mmdet 框架训练的目标分割检测器进行推理。
    https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/1_exist_data_model.md
    '''
    def __init__(self, config_file, checkpoint_file, device = 'cuda:1') -> None:
        '''
        使用 mmdet 框架。
        '''
        self.model = init_detector(config_file, checkpoint_file, device=device)
    
    def detect(self, img):
        ''' 
        暂时只支持 img 为单张图片的路径，或np.ndarray。\n
        返回一个tuple：( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
        '''
        if isinstance(img, str) or isinstance(img, np.ndarray):
            bbox_results, mask_results = inference_detector(self.model, img) # (bbox_results, mask_results) == ( [np.array(100个检测框：[x1,y1,x2,y2,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
            bbox_results[0][:,2] -= bbox_results[0][:,0]  # x1,y1,x2,y2 -> x1,y1,w,h
            bbox_results[0][:,3] -= bbox_results[0][:,1]
            return bbox_results, mask_results   # ( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
        else:
            raise ValueError('not implemented.')

    def summary(self):
        '''
        打印模型结构。
        '''
        print(self.model)

class Detector_mmdet_only_instances(Detector_base):
    '''
    适配mmlab2
    使用 mmdet 框架训练的目标检测检测器进行推理。
    /home/xxy/mmlab2/mmdetection/demo/inference_demo.ipynb
    '''
    def __init__(self, config_file, checkpoint_file, device = 'cuda:1') -> None:
        '''
        使用 mmdet 框架。
        '''
        self.model = init_detector(config_file, checkpoint_file, device=device)
    
    def detect(self, img):
        ''' 
        暂时只支持 img 为单张图片的路径，或np.ndarray。\n
        返回一个tuple：( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
        '''
        if isinstance(img, str) or isinstance(img, np.ndarray):
            if isinstance(img, str):
                img = cv2.imread(img) # (h,w,c)
            # (bbox_results, mask_results) == ( [np.array(100个检测框：[x1,y1,x2,y2,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
            bbox_results = []
            mask_results = [[np.zeros(tuple(img.shape[:2])).astype(bool) for _ in range(100)]]
            result = inference_detector(self.model, img)
            xywh = result.pred_instances.bboxes.cpu().numpy() # 右边是xyxy
            xywh[:,2] -= xywh[:,0]  # x1,y1,x2,y2 -> x1,y1,w,h
            xywh[:,3] -= xywh[:,1]
            bbox_results = np.append(
                xywh,
                result.pred_instances.scores.cpu().numpy().reshape(-1,1),
                axis=1)
            bbox_results = [[r for r in bbox_results]]
            return bbox_results, mask_results   # ( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
        else:
            raise ValueError('not implemented.')


# config_file = "/home/xxy/mmdetection/configs/my_configs/mask2formerBiggestcontour_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
# # config_file = "/home/xxy/mmdetection/configs/my_configs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
# checkpoint_file = "/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth"
config_file = "/home/xxy/mmlab2/mmdetection/work_dirs/faster-rcnn_x101-64x4d_fpn_ms-3x_wailaiwuzhong_500e-shenlan/faster-rcnn_x101-64x4d_fpn_ms-3x_wailaiwuzhong_500e-shenlan.py"
checkpoint_file = "/home/xxy/mmlab2/mmdetection/work_dirs/faster-rcnn_x101-64x4d_fpn_ms-3x_wailaiwuzhong_500e-shenlan/20230616_144549/best_coco_bbox_mAP_epoch_26.pth"
img = 'datasets/shenlan1/30fps300s/0001.jpg'

if __name__ == '__main__':
    register_all_modules()
    # d = Detector_mmdet(config_file, checkpoint_file)
    d = Detector_mmdet_only_instances(config_file, checkpoint_file)
    # d.summary()
    detection = d.detect(img)
    pass
    print(detection)