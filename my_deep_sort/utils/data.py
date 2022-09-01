import os, cv2

class Dataset:
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
            return os.path.join(self.image_folder, self.image_names[index])
        elif index in self.image_names:
            return os.path.join(self.image_folder, index)
        else:
            raise ValueError('index 必须是 ①int:0~len  或  ②图片文件名（不带路径）')

    def __repr__(self) -> str:
        return  f'{self.image_folder}\n' + \
                f'num of images: {self.size}\n'


imgFolder = 'datasets/lingshui/30fps300s'  
if __name__ == '__main__':
    d = Dataset(imgFolder)
    print(d)
    print(len(d))
    print(d[2])
    print(d['0222.jpg'])
    print(d['022s2.jpg'])