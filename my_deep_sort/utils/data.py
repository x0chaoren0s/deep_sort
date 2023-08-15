from cProfile import label
from cgi import test
import os, cv2, random, json
from tkinter import image_types
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
# from torchvision.transform import ToTensor
import numpy as np
try:
    from my_deep_sort.utils.detector import Detector_mmdet
    from my_deep_sort.utils.myTools import *
except:
    try:
        from utils.detector import Detector_mmdet
        from utils.myTools import *
    except:
        from detector import Detector_mmdet
        from myTools import *

class LingshuiFrameDataset(Dataset):
    ''' 索引返回读取到的图片，而不是文件路径 '''
    def __init__(self, image_folder='datasets/lingshui/30fps300s') -> None:
        self.image_folder   = image_folder
        self.image_names    = sorted(os.listdir(self.image_folder))
        self.image_shape    = cv2.imread(os.path.join(self.image_folder,self.image_names[0]), # 目的是获取图片shape，所以只用第一张图就行了
                                cv2.IMREAD_GRAYSCALE).shape
        self.size           = len(self.image_names)
        

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        ''' index: int:0~len  或  图片名（不带路径）'''
        if isinstance(index, int):
            image_path = os.path.join(self.image_folder, self.image_names[index])
        elif index in self.image_names:
            image_path = os.path.join(self.image_folder, index)
        else:
            raise ValueError('index 必须是 ①int:0~len  或  ②图片文件名（不带路径）')
        return cv2.imread(image_path)

    def __repr__(self) -> str:
        return  f'{self.image_folder}\n' + \
                f'num of images: {self.size}\n'
    
class ShenlanFrameDataset(LingshuiFrameDataset):
    ''' 索引返回读取到的图片，而不是文件路径 '''
    def __init__(self, image_folder='datasets/shenlan1/30fps300s') -> None:
        super().__init__(image_folder)

class MOT16TrainFrameDataset(LingshuiFrameDataset):
    ''' 索引返回读取到的图片，而不是文件路径 '''
    sequence_dir = {
        'mot16-02': 'MOT16/train/MOT16-02', 
        'mot16-04': 'MOT16/train/MOT16-04', 
        'mot16-05': 'MOT16/train/MOT16-05', 
        'mot16-09': 'MOT16/train/MOT16-09', 
        'mot16-10': 'MOT16/train/MOT16-10',  
        'mot16-11': 'MOT16/train/MOT16-11', 
        'mot16-13': 'MOT16/train/MOT16-13'
    }
    image_folder = {
        'mot16-02': 'MOT16/train/MOT16-02/img1', 
        'mot16-04': 'MOT16/train/MOT16-04/img1', 
        'mot16-05': 'MOT16/train/MOT16-05/img1', 
        'mot16-09': 'MOT16/train/MOT16-09/img1', 
        'mot16-10': 'MOT16/train/MOT16-10/img1',  
        'mot16-11': 'MOT16/train/MOT16-11/img1', 
        'mot16-13': 'MOT16/train/MOT16-13/img1'
    }
    detection_file = {
        'mot16-02': 'resources/detections/MOT16_POI_train/MOT16-02.npy', 
        'mot16-04': 'resources/detections/MOT16_POI_train/MOT16-04.npy', 
        'mot16-05': 'resources/detections/MOT16_POI_train/MOT16-05.npy', 
        'mot16-09': 'resources/detections/MOT16_POI_train/MOT16-09.npy', 
        'mot16-10': 'resources/detections/MOT16_POI_train/MOT16-10.npy',  
        'mot16-11': 'resources/detections/MOT16_POI_train/MOT16-11.npy', 
        'mot16-13': 'resources/detections/MOT16_POI_train/MOT16-13.npy'
    }
    def __init__(self, args) -> None:
        '''
        Parameters in args
        ----------
        dataset : str in ['mot16-02', 'mot16-04', 'mot16-05', 'mot16-09', 'mot16-10', 'mot16-11', 'mot16-13']
        '''
        self.image_folder   = MOT16TrainFrameDataset.image_folder[args.dataset]
        self.image_names    = sorted(os.listdir(self.image_folder))
        self.image_shape    = cv2.imread(os.path.join(self.image_folder,self.image_names[0]), # 目的是获取图片shape，所以只用第一张图就行了
                                cv2.IMREAD_GRAYSCALE).shape
        self.size           = len(self.image_names)

class LingshuiFramePositivePatchDataset(Dataset):
    def __init__(self, image_path, tlwh_list, patch_resize=(224,224)):
        self.img = cv2.imread(image_path) # (h,w,c)
        self.tlwh_list = [[int(v) for v in tlwh] for tlwh in tlwh_list]
        self.len = len(self.tlwh_list)
        self.patch_resize = patch_resize
    
    def __getitem__(self, index):
        x,y,w,h=self.tlwh_list[index]
        return cv2.resize(self.img[y:y+h,x:x+w], self.patch_resize)

    def __len__(self):
        return self.len

class LingshuiFrameNegativePatchDataset(Dataset):
    def __init__(self, image_path, tlwh_list, patch_resize=(224,224), iou_thre=0.7):
        self.img = cv2.imread(image_path) # (h,w,c)
        self.positive_tlwh_list = [[int(v) for v in tlwh] for tlwh in tlwh_list]
        self.len = len(self.tlwh_list)
        self.patch_resize = patch_resize
        self.iou_thre = iou_thre
        self.negative_tlwh_list = self.get_negative_tlwh_list()

    def get_negative_tlwh_list(self):
        neg = []
        for _ in range(len(self.positive_tlwh_list)):
            w = random.randint(self.patch_resize[0]/2,self.patch_resize[0])
            h = random.randint(self.patch_resize[1]/2,self.patch_resize[1])
            x = random.randint(0,self.img.shape[1]-1-w)
            y = random.randint(0,self.img.shape[0]-1-h)
            while not self.is_negative([x,y,w,h]):
                w = random.randint(self.patch_resize[0]/2,self.patch_resize[0])
                h = random.randint(self.patch_resize[1]/2,self.patch_resize[1])
                x = random.randint(0,self.img.shape[1]-1-w)
                y = random.randint(0,self.img.shape[0]-1-h)
            neg.append([x,y,w,h])
        return neg
            
    def is_negative(self, tlwh):
        for pos in self.positive_tlwh_list:
            if self.iou(tlwh, pos)>=self.iou_thre:
                return False
        return True
    
    def iou(self, tlwh1,tlwh2):
        area1, area2 = tlwh1[2]*tlwh1[3], tlwh2[2]*tlwh2[3]
        x11,y11,x12,y12 = tlwh1[0], tlwh1[1], tlwh1[0]+tlwh1[2], tlwh1[1]+tlwh1[3]
        x21,y21,x22,y22 = tlwh2[0], tlwh2[1], tlwh2[0]+tlwh2[2], tlwh2[1]+tlwh2[3] 
        x31,y31,x32,y32 = max(x11,x21), max(y11,y21), min(x12,x22), min(y12,y22)
        w3 = x32-x31 if x32>x31 else 0
        h3 = y32-y31 if y32>y31 else 0
        area3 = w3*h3
        return 1.0 * area3 / (area1 + area2 - area3)
    
    def __getitem__(self, index):
        x,y,w,h=self.negative_tlwh_list[index]
        return cv2.resize(self.img[y:y+h,x:x+w], self.patch_resize)

    def __len__(self):
        return self.len

def iou(tlwh1,tlwh2):
    area1, area2 = tlwh1[2]*tlwh1[3], tlwh2[2]*tlwh2[3]
    x11,y11,x12,y12 = tlwh1[0], tlwh1[1], tlwh1[0]+tlwh1[2], tlwh1[1]+tlwh1[3]
    x21,y21,x22,y22 = tlwh2[0], tlwh2[1], tlwh2[0]+tlwh2[2], tlwh2[1]+tlwh2[3] 
    x31,y31,x32,y32 = max(x11,x21), max(y11,y21), min(x12,x22), min(y12,y22)
    w3 = x32-x31 if x32>x31 else 0
    h3 = y32-y31 if y32>y31 else 0
    area3 = w3*h3
    return 1.0 * area3 / (area1 + area2 - area3)

def buildLingshuiFramePatchDataset(frameFolder, outputFolder, confi_thre=0.7, bbox_move=0.2):
    ''' 使用图像识别检测器的检测数据 '''
    config_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    checkpoint_file = '/home/xxy/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_FlatCosineAnnealing/epoch_2000.pth'
    detector = Detector_mmdet(config_file, checkpoint_file)

    os.makedirs(outputFolder, exist_ok=True)
    
    testDataLoader = DataLoader(dataset=LingshuiFrameDataset(frameFolder))
    patch_id = 0
    for img in testDataLoader: # img: torch.Tensor   shape: (1, 1080, 1920, 3)
        img = img[0].numpy()   # img: numpy.ndarray  shape: (1080, 1920, 3)
        detections = detector.detect(img) # 一个tuple：( [np.array(100个检测框：[x1,y1,w,h,confidence],[],..)], [[100个mask：np.array(..),np.array()，..]] )
        for d in detections[0][0]:
            if d[4]<confi_thre:
                continue
            x,y,w,h = d[:4]
            # bbox random move
            dw, dh = int(w*bbox_move), int(h*bbox_move)
            w = int(min(img.shape[1],max(0,w+random.randint(-dw, dw))))  # img.shape: (h,w,c)
            h = int(min(img.shape[0],max(0,h+random.randint(-dh, dh))))  # img.shape: (h,w,c)
            x = int(min(img.shape[1]-w,max(0,x+random.randint(-dw, dw))))
            y = int(min(img.shape[0]-h,max(0,y+random.randint(-dh, dh))))
            patch = img[y:y+h,x:x+w]
            patch_file = os.path.join(outputFolder, '%07d.jpg'%patch_id)
            cv2.imwrite(patch_file, patch)
            patch_id += 1

def buildLingshuiFramePatchDataset2(labelmeJsonFolder, outputFolder, bbox_move=0.2):
    ''' 使用原始手工标注数据 '''
    pos_folder = os.path.join(outputFolder,'positive')
    neg_folder = os.path.join(outputFolder,'negative')
    os.makedirs(pos_folder, exist_ok=True)
    os.makedirs(neg_folder, exist_ok=True)
    patch_id = 0
    for file in os.listdir(labelmeJsonFolder):
        with open(os.path.join(labelmeJsonFolder,file), 'r') as f:
            jsondict = json.load(f)
        image_path = os.path.join(labelmeJsonFolder, linux_path(jsondict['imagePath']))
        img = cv2.imread(image_path)
        shapes = jsondict['shapes']
        def shape2tlwh(shape):
            # labelme：[topLeft.x, topLeft.y, downRigth.x, downRight.y]
            # cv2 ：[topLeft.x, topLeft.y, width, hight]
            points = shape['points']
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            x,y,x2,y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            w, h = x2-x, y2-y
            return [x,y,w,h]
        pos_tlwhs = [shape2tlwh(s) for s in shapes]
        pos_tlwhs = [tlwh for tlwh in pos_tlwhs if tlwh[2]>0 and tlwh[3]>0]
        # get_negative_tlwh_list
        neg_tlwhs = []
        iou_thre = 0.8
        def is_negative(tlwh):
            for pos in pos_tlwhs:
                if iou(tlwh, pos)>=iou_thre:
                    return False
            return True
        for pos_tlwh in pos_tlwhs:
            pos_size = pos_tlwh[2:]
            w = random.randint(int(pos_size[0]/2),int(pos_size[0]))
            h = random.randint(int(pos_size[1]/2),int(pos_size[1]))
            x = random.randint(0,int(img.shape[1]-1-w))
            y = random.randint(0,int(img.shape[0]-1-h))
            while not is_negative([x,y,w,h]):
                w = random.randint(int(pos_size[0]/2),int(pos_size[0]))
                h = random.randint(int(pos_size[1]/2),int(pos_size[1]))
                x = random.randint(0,int(img.shape[1]-1-w))
                y = random.randint(0,int(img.shape[0]-1-h))
            neg_tlwhs.append([x,y,w,h])

        for i,tlwh in enumerate(pos_tlwhs):
            x,y,w,h = tlwh[:4]
            patch = img[y:y+h,x:x+w]
            patch_file = os.path.join(pos_folder, '%07d.jpg'%patch_id)
            cv2.imwrite(patch_file, patch)
            x,y,w,h = neg_tlwhs[i][:4]
            patch = img[y:y+h,x:x+w]
            patch_file = os.path.join(neg_folder, '%07d.jpg'%patch_id)
            cv2.imwrite(patch_file, patch)
            patch_id += 1

class LingshuiFramePatchDataset(Dataset):
    def __init__(self, patchFolder, patch_resize=(224,224), device='cuda:1') -> None:
        '''
        patchFolder 内部需要包含 positive 和 negative 两个文件夹，分别存放(不一定等量)正负样本的 frame-patch 图片 \n
        返回的 dataset 前半部分是正样本，后半部分是负样本，dataset.len 是正负样本总和 \n
        后续 dataloader 必须做 shuffle \n
        对返回的 dataset 做索引，返回resize后的 frame-patch 图片和对应的标签（正样本0.98，负样本0.02）
        '''
        self.patchFolder = patchFolder
        self.patch_resize = patch_resize
        self.posFolder = os.path.join(self.patchFolder,'positive')
        self.negFolder = os.path.join(self.patchFolder,'negative')
        self.posPatchNames = sorted(os.listdir(self.posFolder))
        self.negPatchNames = sorted(os.listdir(self.negFolder))
        self.patchNames = self.posPatchNames + self.negPatchNames
        self.poslen = len(os.listdir(self.posFolder))
        self.neglen = len(os.listdir(self.negFolder))
        self.len = self.poslen + self.neglen
        self.device = device


    def __getitem__(self, index):
        imgPath = os.path.join(self.posFolder, self.patchNames[index]) if index<self.poslen \
                else os.path.join(self.negFolder, self.patchNames[index])
        img = cv2.imread(imgPath)
        img = cv2.resize(img, self.patch_resize)
        img = Tensor(img).float()
        img = img.permute(2,0,1)
        tag = [0.98] if index < self.poslen else [0.02]
        tag = Tensor(tag).float()
        return img.to(self.device), tag.to(self.device)

    
    def __len__(self):
        return self.len
        
    
labelmeJsonFolder = '/home/xxy/deep_sort/datasets/2021-03-06-09-52-50/jsonsnew'
imgFolder = '/home/xxy/deep_sort/datasets/lingshui/30fps300s'
patchFolder = '/home/xxy/deep_sort/datasets/lingshui/patches'  
patchFolder2 = '/home/xxy/deep_sort/datasets/lingshui/patches2'  
if __name__ == '__main__':
    pass
    # d = Dataset(imgFolder)
    # print(d)
    # print(len(d))
    # print(d[2])
    # print(d['0222.jpg'])
    # print(d['022s2.jpg'])
    # buildLingshuiFramePatchDataset(imgFolder, patchFolder, 0.9, 0)
    # buildLingshuiFramePatchDataset2(labelmeJsonFolder, patchFolder2, 0)
    lfd = LingshuiFramePatchDataset(patchFolder2)
    print(66)