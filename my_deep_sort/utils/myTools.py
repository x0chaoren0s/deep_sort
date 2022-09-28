import numpy as np
import cv2 as cv
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import time,math,cv2,os

def show(img,fixedRange=False,figsize=(20,20)):
    plt.figure(figsize=figsize)
    if img.ndim==3:
        plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    else:
        if fixedRange:
            plt.imshow(img,cmap='gray',vmin=0,vmax=255)
        else:
            plt.imshow(img,cmap='gray')
    plt.show()
def showList(img_list, columns=3):
    for i in range(0,len(img_list),columns):
        batchlist=img_list[i:min(i+columns,len(img_list))]
        if sum([len(img.shape) for img in batchlist])/len(batchlist) not in [2,3]:
            #必须全是灰度图或者全是彩图
            batchlist=[img if len(img.shape)==3 else cv.cvtColor(img,cv.COLOR_GRAY2BGR)
                      for img in batchlist]
        show(np.hstack(batchlist))
def showList2(img_list):
    imgInd=0
    while 0<=imgInd and imgInd<len(img_list):
        cv.imshow('img',img_list[imgInd])
        key=cv.waitKey(0)
        if key==ord('q'):
            break
        if key==ord('a') or key==ord('w'):
            imgInd-=1
        else:
            imgInd+=1
    cv.destroyAllWindows()
    
def rectangle(ndarray_bin, tlwh, color=bool(1), thickness=1):
    x1,y1,w,h = tlwh[:4]
    pt1, pt2 = (int(x1),int(y1)), (int(x1+w),int(y1+h))
    return cv2.rectangle(ndarray_bin,pt1,pt2,color,thickness)

def pltfig2cvimg(pltfig, dpi=1000):
    # define a function which returns an image as numpy array from figure
    #https://blog.csdn.net/weixin_39856265/article/details/110701257
    buf = io.BytesIO()
    pltfig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img
def grayhist(img,dstsize=(1000,700)):
    if len(img.shape)==3:
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #使用agg后plt绘图就不会显示出来，但会UserWarning，因此用ignore关闭warning
    warnings.filterwarnings("ignore")
    matplotlib.use('agg')
    plt.ion()
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(img.ravel(),256,[0,256])
    plt.show()
    img=pltfig2cvimg(fig,100)
    
    plt.ioff()
    return img


# img=np.arange(15).reshape((3,5))
def splitImg(img, rows=3, colunms=2):
    subImgHeight,subImgWidth=round(img.shape[0]/rows),round(img.shape[1]/colunms)
    #所有子图的第一行序号，最后一个是哨兵
    rowBegs=[i for i in range(0,img.shape[0],subImgHeight)][:rows]+[img.shape[0]]
    #所有子图的第一列序号，最后一个是哨兵
    colBegs=[i for i in range(0,img.shape[1],subImgWidth)][:colunms]+[img.shape[1]]
    subImgs=[]
    for ri in range(rows):
        for ci in range(colunms):
            subImgs.append(img[rowBegs[ri]:rowBegs[ri+1],colBegs[ci]:colBegs[ci+1]])
    return subImgs
# img,type(img),splitImg(img)
# splitedPixelRanges=[[] for (r,c) in splitRowsColumns]
# splitedWholeFrames=[[] for (r,c) in splitRowsColumns]
# splitedPixelRanges


def combineSubImg(wholeImg2DShape,numSubimgByRow,numSubimgByCol,subImgListByRow):
    wholeImgShape=(*wholeImg2DShape,3)
    wholeImg=np.zeros(wholeImgShape)
    wholer=0
    for ri in range(numSubimgByRow):
        wholec=0
        for ci in range(numSubimgByCol):
            k=ci+ri*numSubimgByCol
            subImg=subImgListByRow[k].astype(np.uint8)
            subr,subc=subImg.shape[:2]
            if len(subImgListByRow[k].shape)==2:
                subImg=cv.cvtColor(subImg,cv.COLOR_GRAY2BGR)
            print(subImgListByRow[k].shape,subImg.shape)
            wholeImg[wholer:wholer+subr,wholec:wholec+subc]=subImg
            wholec+=subc
        wholer+=subr
    return wholeImg.astype(np.uint8)
# print(img,splitImg(img),combineSubImg((3,5),3,2,splitImg(img)))
# show(img)
# show(combineSubImg((3,5),3,2,splitImg(img)))

def linux_path(path: str) -> str:
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)