# https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/1_exist_data_model.md

from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np

class Detector:
    def __init__(self, config_file, checkpoint_file, device = 'cuda:1') -> None:
        self.config_file        = config_file
        self.checkpoint_file    = checkpoint_file
        self.device             = device
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
    
    def detect(self, img):
        ''' 
        暂时只支持 img 为单张图片的路径，或np.ndarray。\n
        返回一个tuple：( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
        '''
        if isinstance(img, str) or isinstance(img, np.ndarray):
            return inference_detector(self.model, img)
        else:
            raise ValueError('not implemented.')

# 测试单张图片并展示结果
# img = 'datasets/lingshui/30fps300s/0001.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# img = mmcv.imread(img)
# result = inference_detector(model, img) # 是一个tuple：( [np.array(100个检测框：[x1,y1,w,h,p],[],..)],  [[100个mask：np.array(..),np.array()，..]]  )
# inference_detector(model, imgs) 返回：
# If imgs is a list or tuple, the same length list type results
# will be returned, otherwise return the detection results directly.


# 在一个新的窗口中将结果可视化
# model.show_result(img, result)
# 或者将可视化结果保存为图片
# model.show_result(img, result, out_file='tmp/result.jpg')

# # 测试视频并展示结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)