# https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/1_exist_data_model.md

from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np

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
    使用 mmdet 框架训练的检测器进行推理。
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


config_file = "/home/xxy/mmdetection/configs/my_configs/mask2formerBiggestcontour_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
# config_file = "/home/xxy/mmdetection/configs/my_configs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
checkpoint_file = "/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth"

if __name__ == '__main__':
    d = Detector_mmdet(config_file, checkpoint_file)
    d.summary()