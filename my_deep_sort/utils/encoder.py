from sys import prefix
from turtle import forward
import numpy as np
import torchvision.models as models
from torch import nn
import torch
from tqdm import tqdm
try:
    from data import LingshuiFramePatchDataset
except:
    from my_deep_sort.utils.data import LingshuiFramePatchDataset
from torch.utils.data import DataLoader
import os, cv2
from torch.utils.tensorboard import SummaryWriter

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        res18_128 = models.resnet18()       # 原来是输出1000维
        res18_128.fc = nn.Linear(512, 128)  # 现在改成输出128维
        self.backbone = res18_128
        self.head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.squeeze(-1)
        feature = self.backbone(input)
        feature = torch.flatten(feature, 1) #拉成二维向量[batch_size, size]
        output = self.head(feature)
        output = output.squeeze(-1) # torch.Size([256]) 变成 torch.Size([256, 1]) 和输入匹配
        return output, feature

class ApparentFeatureExtracker:
    ''' 用于抽取检测框的表观特征 '''
    def __init__(self, checkpoint_file=None, device = 'cuda:1', patch_resize=(224,224)) -> None:
        self.model = Net()
        self.checkpoint_file = checkpoint_file
        if self.checkpoint_file is not None:
            self.load_state(self.checkpoint_file)
        self.device = device
        self.model = self.model.to(self.device)
        self.patch_resize = patch_resize

    def extract_feature(self, img, tlwh):  # img 实际传进来的是 cv2 的 image：(1080, 1920, 3)==(h, w, c)
        '''
        img: 单张图片的路径，或np.ndarray。\n
        tlwh: [x1,y1,w,h] 或 [x1,y1,w,h,confidence]
        '''
        x,y,w,h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])    # img 要转化成 patch: torch.Size([1, 3, 224, 224])
        if h==0 or w==0:
            return np.random.random(128).astype(np.float32)        # shape: (128,)
        patch = cv2.resize(img[y:y+h, x:x+w], self.patch_resize)            # patch: (224, 224, 3)
        patch = torch.Tensor(patch).float()                                 # patch: torch.Size([224, 224, 3])
        patch = patch.permute(2,0,1)                                        # patch: torch.Size([3, 224, 224])
        patch = patch.reshape(1, *patch.shape).to(self.device)              # patch: torch.Size([1, 3, 224, 224])

        self.model.eval()
        prediction, feature = self.model(patch) # feature: torch.Size([1, 128]) torch.float32
        return feature[0].cpu().detach().numpy()   # shape: (128,)  dtype=float32

    def train(self, epochs, patchFolder, train_in_whole, batch_size, checkpoint_folder, checkpoint_freq, checkpoint_prefix, tensorboardLog_folder):
        '''
        @params
        epochs: 完整训练一次所有训练数据的次数 \n
        patchFolder: 内部需要包含 positive 和 negative 两个文件夹，分别存放(不一定等量)正负样本的 frame-patch 图片 \n
        train_in_whole: 训练集占整体数据集(patchFolder)的比例 \n
        batch_size: 训练或验证时载入显存的批量大小 \n
        checkpoint_folder: 权重文件保存地址 \n
        checkpoint_freq: 权重文件保存间隔，即每训练多少个epoch保存一次 \n
        checkpoint_prefix: 如果存在，权重文件名为 {checkpoint_prefix}_epoch{checkpoint_freq * times}.pth ，否则为 epoch{checkpoint_freq * times}.pth \n
        tensorboardLog_folder: 用于调用 tensorboard 实时监测训练过程 \n
        '''
        whole_set = LingshuiFramePatchDataset(patchFolder, device=self.device)
        train_size = int(train_in_whole * len(whole_set))
        valid_size = len(whole_set) - train_size
        train_set, valid_set = torch.utils.data.random_split(whole_set, [train_size, valid_size])

        trainDataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validDataLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        
        loss_fun = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(self.model.parameters())

        writer = SummaryWriter(tensorboardLog_folder)
        if checkpoint_prefix is None:
            checkpoint_prefix = ''
        else:
            checkpoint_prefix = checkpoint_prefix + '_'
        
        for epoch in range(1,epochs + 1):
            processBar = tqdm(trainDataLoader) # 只对train读条，但最后同时展示本epoch的train和valid的loss和acc

            self.model.train()
            for batch, (trainImgs, labels) in enumerate(processBar):
                # self.model.zero_grad() # 和 optimizer.zero_grad() 选一个使用
                optimizer.zero_grad()
                predictions, features = self.model(trainImgs)   # trainImgs: torch.Size([32, 3, 224, 224])
                # print(predictions)
                labels = labels.squeeze(-1) # 自带smoth，正0.98，负0.02
                loss = loss_fun(predictions,labels)
                accuracy = torch.sum(abs(predictions - labels)<0.02)/labels.shape[0]
                loss.backward()

                optimizer.step()
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                        (epoch,epochs,loss.item(),accuracy.item()))

                # 一个epoch训练完成后做验证，为了继续使用processBar所以写在循环内部
                if batch == len(processBar)-1:
                    self.model.eval()
                    correct, validLoss = torch.tensor([0.],device=self.device), torch.tensor([0.],device=self.device)
                    self.model.eval()
                    with torch.no_grad():
                        for validImgs,labels in validDataLoader:
                            labels = labels.squeeze(-1)
                            predictions, features = self.model(validImgs)
                            loss = loss_fun(predictions,labels)
                            
                            validLoss += loss
                            correct += torch.sum(abs(predictions - labels)<0.02)
                    validAccuracy = correct/len(valid_set)
                    validLoss = validLoss/len(valid_set)
                    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                        (epoch,epochs,loss.item(),accuracy.item(), validLoss.item(), validAccuracy.item()))
            # tensorboard 可视化本 epoch 的 loss 和 acc
            writer.add_scalar('train loss', loss.item(), epoch)
            writer.add_scalar('train acc', accuracy.item(), epoch)
            writer.add_scalar('valid loss', validLoss.item(), epoch)
            writer.add_scalar('valid acc', validAccuracy.item(), epoch)
            # 展示验证过程的最后一个 batch 的图片和模型预测结果。   暂未实现
            # for img in validImgs:
            # writer.add_image_with_boxes('valid_image', validImgs[-1], [0,0,224,224], epoch, labels=[])

            # 保存权重文件
            if epoch % checkpoint_freq == 0:
                os.makedirs(checkpoint_folder, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_folder, f'{checkpoint_prefix}epoch{epoch}.pth')

                torch.save(self.model.state_dict(), checkpoint_file)

    def load_state(self, checkpoint_file):
        assert os.path.exists(checkpoint_file)
        self.model.load_state_dict(torch.load(checkpoint_file))
            

class ApparentFeatureFakeCopier:
    '''
    直接循环复制 resources/detections/MOT16_POI_test/MOT16-06.npy 中的特征。
    该特征并不能针对实际需要抽取特征的检测框，因此是fake。
    '''
    def __init__(self, _) -> None:
        self.input_file = 'resources/detections/MOT16_POI_test/MOT16-06.npy'
        self.raw_features = np.load(self.input_file)[:,-128:]
        self.len = len(self.raw_features)
        self.idx = 0

    def extract_feature(self, img, tlwh):
        '''
        img 为单张图片的路径，或np.ndarray。\n
        直接循环返回 resources/detections/MOT16_POI_test/MOT16-06.npy 中的特征
        '''
        feature = self.raw_features[self.idx]
        self.idx = (self.idx+1 )%self.len
        return feature
        
class ApparentFeatureCopier:
    '''
    抽取别人训练好的 resources/detections/MOT16_POI_train 中的特征。
    只能用于 MOT16/train 数据集。（test没有gt数据因此不用）
    '''
    def __init__(self, input_file='resources/detections/MOT16_POI_test/MOT16-06.npy') -> None:
        '''
        input_file: 别人抽取好特征的.npy文件
        '''
        # self.input_file = 'resources/detections/MOT16_POI_test/MOT16-06.npy'
        self.raw_features = np.load(input_file)[:,-128:]
        self.len = len(self.raw_features)
        self.idx = 0

    def extract_feature(self, img, tlwh):
        '''
        img 为单张图片的路径，或np.ndarray。\n
        直接循环返回 resources/detections/MOT16_POI_test/MOT16-06.npy 中的特征
        '''
        feature = self.raw_features[self.idx]
        self.idx = (self.idx+1 )%self.len
        return feature

checkpoint_file = '/home/xxy/deep_sort/my_deep_sort/detector_checkpoints/renet18/epoch300.pth'
checkpoint_folder = '/home/xxy/deep_sort/my_deep_sort/detector_checkpoints/renet18'
tensorboardLog_folder = '/home/xxy/deep_sort/my_deep_sort/detector_checkpoints/renet18/log'
patchFolder2 = '/home/xxy/deep_sort/datasets/lingshui/patches2' 
if __name__ == '__main__':
    # n = Net()
    # print(n)
    a = ApparentFeatureExtracker(checkpoint_file)
    a.train(epochs=60, patchFolder=patchFolder2, train_in_whole=0.8, batch_size=32, 
        checkpoint_folder=checkpoint_folder, checkpoint_freq=30, checkpoint_prefix='base300epoch',
        tensorboardLog_folder=tensorboardLog_folder)
