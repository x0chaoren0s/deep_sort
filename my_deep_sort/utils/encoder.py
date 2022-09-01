import numpy as np

class Encoder:
    ''' 用于抽取检测框的表观特征 '''
    def __init__(self, config_file, checkpoint_file, device = 'cuda:1') -> None:
        self.model = '还没实现'

    def extract_feature(self, img, tlwh):
        '''
        暂时只支持 img 为单张图片的路径。\n
        还没实现model，因此此处返回一个128维随机向量
        '''
        return np.random.random(128)