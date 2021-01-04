import cv2 as cv
import numpy as np
import math
import copy

#   @fn                             绘制直方图
#   @param  histArray               输入hist矩阵，每一行为1个通道的hist
#   @param  height                  直方图的高度
#   @param  grayLvLen               每个灰度级在hist中的宽度，grayLvLen == 2时，对于256灰度级，直方图宽度为512
#   @param  name                    显示窗口的名称
#   @param                          void
def drawHist(histArray,height,grayLvLen,name):
    channels = histArray.shape[0]
    data = np.zeros((channels,256*grayLvLen),dtype = 'float32')
    k = np.array([0,0,0],dtype = 'float32')
    # 缩放系数
    for i in range(0,channels):
        k[i] = height  / (max(histArray[i])) * 0.8
    # 缩放后的数据
    for i in range(0,channels):
        for j in range(0,256*grayLvLen):
            data[i][j] = round(histArray[i][j//grayLvLen] * k[i])
    # draw
    for i in range(0,channels):
        img = np.zeros((height,256*grayLvLen,channels),dtype = 'uint8')
        for j in range(0,256*grayLvLen):
            img[height-int(data[i][j]):height-1,j,i] = 255
        cv.imshow(name+" channels - "+str(i),img)


#   @fn                             获取histArray
#   @param  src                     输入图像
#   @return                         hist矩阵
def getHist(src):
    rows = src.shape[0]
    cols = src.shape[1]
    if(len(src.shape) == 2):
        channels = 1
    else:
        channels = src.shape[2]
    dst = np.zeros((channels,256),dtype = 'float32')
    if(channels > 1):
        for i in src:
            for j in i:
                c = 0
                for k in j:
                    dst[c][k] += 1
                    c += 1
    else:
        for i in src:
            for j in i:
                dst[0][j] += 1
    return dst

#   @fn                             直方图均衡化
#   @param  src                     输入图像
#   @return                         直方图均衡化后的图像
def histogramEqualization(src):
    srcRow = src.shape[0]
    srcCol = src.shape[1]
    hist = getHist(src)
    channels = hist.shape[0]
    pixelCnt = src.shape[0] * src.shape[1]
    Pr = copy.deepcopy(hist)
    # 计算映射
    for i in range(0,channels):
        for j in range(0,256):
            Pr[i][j] = Pr[i][j] / pixelCnt * 255
    for i in range(0,channels):
        for j in range(1,256):
            Pr[i][j] = Pr[i][j] + Pr[i][j-1]
    for i in range(0,channels):
        for j in range(0,256):
            Pr[i][j] = round(Pr[i][j])
    dst = np.zeros(src.shape,src.dtype)
    # 映射
    if(channels == 1):
        for i in range(0,srcRow):
            for j in range(0,srcCol):
                dst[i][j] = Pr[0][src[i][j]]
    else:
        for i in range(0,srcRow):
            for j in range(0,srcCol):
                for k in range(0,channels):
                    dst[i][j][k] = Pr[k][src[i][j][k]]
    # 计算处映射后图像的hist
    dstHist = np.zeros(hist.shape,hist.dtype)
    for i in range(0,channels):
        for j in range(0,256):
            dstHist[i][int(Pr[i][j])] += hist[i][j]
    # draw hist
    drawHist(hist,512,2,"origin hist")
    drawHist(dstHist,512,2,"processed hist")
    return dst

input = cv.imread("F://lena.jpg")
#input = cv.cvtColor(input,cv.COLOR_RGB2GRAY)
res = histogramEqualization(input)
cv.imshow("input",input)
cv.imshow("res",res)
cv.waitKey(0)
