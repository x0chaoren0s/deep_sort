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
            return inference_detector(self.model, img)
        else:
            raise ValueError('not implemented.')

