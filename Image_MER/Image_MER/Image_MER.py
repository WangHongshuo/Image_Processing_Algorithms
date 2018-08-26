import cv2 as cv
import numpy as np
import copy
import math
from collections import OrderedDict

#   @class                              二维点
#   @member x                           x坐标
#   @member y                           y坐标
class Point2:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __sub__(self,p):
        res = Point2(-1,-1)
        res.x = self.x - p.x
        res.y = self.y - p.y
        return res
    #   @fn                             print输出点坐标
    #   return                          str
    def __str__(self):
        return "[" + str(self.x) + ", " + str(self.y) + "] "
    #   @fn                             计算两个向量叉积
    #   @param  p                       第二个向量
    #   @return                         叉积
    def crossProduct(self,p):
        return self.x * p.y - self.y * p.x

#   @class                              边缘信息
#   @member topLeft                     边缘区域左上坐标
#   @member bottomRight                 边缘区域右下坐标
#   @member borderPoints                边缘点集
class Border:
    def __init__(self):
        self.topLeft = Point2(-1,-1)
        self.bottomRight = Point2(-1,-1)
        self.borderPoints = OrderedDict()

#   @fn                                 获取边缘图像
#   @param  src                         输入图像
#   @param  mark                        根据灰度标记获取边缘
#   @return                             边缘图像
def getBinaryImageBorder(src,mark):
    borderImg = copy.deepcopy(src)
    rows = src.shape[0]
    cols = src.shape[1]
    # (x,y)的8邻域全等于标记值，则f(x,y) = 0
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if(src[i][j] == mark and
               src[i-1][j-1] == mark and src[i-1][j] == mark and
               src[i-1][j+1] == mark and src[i][j-1] == mark and
               src[i][j+1] == mark and src[i+1][j-1] == mark and
               src[i+1][j] == mark and src[i+1][j+1] == mark):
                borderImg[i][j] = 0
    return borderImg

#   @fn                                 获取边缘图像的边缘信息
#   @param  src                         输入边缘图像
#   @param  mark                        边缘标记灰度值
#   @return                             边缘信息
def getBorderInfo(src,mark):
    b = Border()
    rows = src.shape[0]
    cols = src.shape[1]
    # 按照从上到下从左到右的顺序录入边缘额点
    for i in range(0,rows):
        for j in range(0,cols):
            if(src[i][j] == mark):
                if(not i in b.borderPoints):
                    b.borderPoints[i] = list()
                b.borderPoints[i].append(Point2(i,j))
    # 获取边缘区域左上右下坐标，边缘全包含在以这两点为顶点的矩形内
    b.topLeft.x = next(iter(b.borderPoints.items()))[0]
    b.bottomRight.x = next(reversed(b.borderPoints.items()))[0]
    b.topLeft.y = (next(iter(b.borderPoints.items()))[1])[0].y
    b.bottomRight.y = (next(reversed(b.borderPoints.items()))[1])[-1].y

    for i in b.borderPoints:
        if(b.borderPoints[i][0].y < b.topLeft.y):
            b.topLeft.y = b.borderPoints[i][0].y
        if(b.borderPoints[i][-1].y > b.bottomRight.y):
            b.bottomRight.y = b.borderPoints[i][-1].y
    return b

#   @fn                                 获取凸包点集
#   @param  bInfo                       目标边缘信息
#   @return                             凸包点集
def getConvexHull(bInfo):
    # dict浅拷贝
    bP = bInfo.borderPoints
    top = bInfo.topLeft.x
    bottom = bInfo.bottomRight.x
    # 从下方开始进行Graham扫描，水平序
    # opencv坐标系对Graham扫描无影响，不比转换坐标系
    pStack = list()
    it = reversed(bP)
    key = next(it)
    pStack.append(bP[key][0])
    pStack.append(bP[key][-1])
    key = next(it)
    pStack.append(bP[key][-1])
    while(key != top):
        key = next(it)
        nextPoint = bP[key][-1]
        vec1 = pStack[-2] - pStack[-1]
        vec2 = nextPoint - pStack[-1]
        cP = vec1.crossProduct(vec2)
        while( cP > 0):
            pStack.pop(-1)
            vec1 = pStack[-2] - pStack[-1]
            vec2 = nextPoint - pStack[-1]
            cP = vec1.crossProduct(vec2)
        pStack.append(nextPoint)
      
    it = iter(bP)
    key = next(it)
    pStack.append(bP[key][0])
    while(key < bottom):
        key = next(it)
        nextPoint = bP[key][0]
        vec1 = pStack[-2] - pStack[-1]
        vec2 = nextPoint - pStack[-1]
        cP = vec1.crossProduct(vec2)
        while( cP > 0):
            pStack.pop(-1)
            vec1 = pStack[-2] - pStack[-1]
            vec2 = nextPoint - pStack[-1]
            cP = vec1.crossProduct(vec2)
        pStack.append(nextPoint)

    # 清除在同一直线上的重复点
    res = list()
    res.append(pStack[0])
    res.append(pStack[1])
    i = 1
    j = 2
    while(j < len(pStack)-1):
        if(res[i].y != pStack[j].y or pStack[j].x == top or pStack[j].x == bottom):
            res.append(pStack[j])
            i += 1
        elif(pStack[j].y != pStack[j+1].y):
            res.append(pStack[j])
            i += 1
        j += 1
    res.append(pStack[-1])
    return res


def getMER(src):
    # 根据标记获取边缘
    borderImg = getBinaryImageBorder(src,255)
    # 根据标记获取边缘点
    b = getBorderInfo(borderImg,255)
    # 获取凸包点集
    ch = getConvexHull(b)
    # 画凸包，绿线为凸包，红点为凸包点
    borderImg = cv.cvtColor(borderImg,cv.COLOR_GRAY2RGB)
    for i in range(1,len(ch)):
        cv.line(borderImg,(ch[i-1].y,ch[i-1].x),(ch[i].y,ch[i].x),(0,255,0),1)
        cv.circle(borderImg,(ch[i].y,ch[i].x),1,(0,0,255),1)
    cv.imshow("Convex Hull",borderImg)
    return ch


input = cv.imread("F://MBR.bmp",cv.IMREAD_GRAYSCALE)
cv.imshow("input",input)
res = getMER(input)
cv.waitKey(0)
