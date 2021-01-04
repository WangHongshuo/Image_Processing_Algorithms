import cv2 as cv
import numpy as np
import copy
import math
from collections import OrderedDict

#   @class                              二维点
#   @member _x                          x坐标
#   @member _y                          y坐标
class Point2:

    def __init__(self,_x,_y):
        self.x = _x
        self.y = _y

    def __sub__(self,p):
        res = Point2(0,0)
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

    #   @fn                             计算两个向量点积
    #   @param  p                       第二个向量
    #   @return                         点积
    def dotProduct(self,p):
        return self.x * p.x + self.y * p.y

class Rect2:
    def __init__(self,_p,_minBorder,_maxBorder,_area):
        self.p = _p
        self.minBorder = _minBorder
        self.maxBorder = _maxBorder
        self.area = _area

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
def getConvexHull(borderInfo):
    # dict浅拷贝
    bP = borderInfo.borderPoints
    top = borderInfo.topLeft.x
    bottom = borderInfo.bottomRight.x
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

#   @fn                                 获取点在直线y=kx+b上投影坐标
#   @param  k                           y=kx+b中的k
#   @param  b                           y=kx+b中的b
#   @param  x                           点坐标x
#   @param  y                           点坐标y
#   @return                             Point2(x,y)，点在直线y=kx+b上投影坐标
def getPointProjectionInLine(k,b,x,y):
    x = (k * (y - b) + x) / (k * k + 1)
    y = k * x + b
    return Point2(x,y)

#   @fn                                 由旋转卡壳法获得的最小外接矩形的5个求矩形4个顶点坐标
#   @param  bottom1                     最小外接矩形底部边上的点1
#   @param  bottom2                     最小外接矩形底部边上的点2
#   @param  top                         最小外接矩形顶部边上的点
#   @param  left                        最小外接矩形左侧边上的点
#   @param  right                       最小外接矩形右侧边上的点
#   @return                             矩形4个顶点坐标
def getRectInfo(bottom1,bottom2,top,left,right):
    p = list([Point2(0,0),Point2(0,0),Point2(0,0),Point2(0,0)])
    if(bottom1.x == bottom2.x):
        p[0] = Point2(bottom1.x,right.y)
        p[1] = Point2(bottom1.x,left.y)
        p[2] = Point2(top.x,left.y)
        p[3] = Point2(top.x,right.y)
    elif(bottom1.y == bottom2.y):
        p[0] = Point2(right.x,bottom1.y)
        p[1] = Point2(left.x,bottom1.y)
        p[2] = Point2(left.x,top.y)
        p[3] = Point2(right.x,top.y)
    else:
        k1 = (bottom1.y - bottom2.y) / (bottom1.x - bottom2.x)
        b1 = bottom2.y - k1 * bottom2.x
        p[0] = getPointProjectionInLine(k1,b1,right.x,right.y)
        p[1] = getPointProjectionInLine(k1,b1,left.x,left.y)
        k2 = k1
        b2 = top.y - k2 * top.x
        p[2] = getPointProjectionInLine(k2,b2,left.x,left.y)
        p[3] = getPointProjectionInLine(k2,b2,right.x,right.y)

    return p

#   @fn                                 由旋转卡壳法求最小外接矩形
#   @param  convexHullPoints            凸包点急
#   @return                             最小外接矩形的信息（4个顶点，高，宽，面积）
def getMinRectByRotatingCalipers(convexHullPoints):
    # convexHullPoints[0]和convexHullPoints[-1]相等
    # 避免在搜索顶点时影响结果，去掉最后一个，搜索用cHP1
    cHP = convexHullPoints
    cHP1 = cHP[0:-1]
    pCnt = len(cHP1)
    # 点少于3个时没有做相关处理
    if(pCnt < 3):
        return -1
    # 初始搜索参数
    # t - 顶部
    # r - 右侧
    # l - 左侧
    t = 2
    r = 2
    l = pCnt - 1
    # 暂存最小参数
    minArea = 0
    minT = 0
    minR = 0
    minL = 0
    minI = 0
    minH = 0
    minW = 0
    for i in range(0,pCnt):
        # 底部向量，以该向量为底，用向量叉积来寻找凸包上距离该向量最远的点t
        # 以t为中间点，用向量的点积寻找t最右边的点r和最左边的点l（投影法）
        vBottom = cHP[i+1] - cHP[i]

        # 顶点t
        vTop = cHP1[t] - cHP[i]
        last = vBottom.crossProduct(vTop)
        curr = 0.0
        while(1):
            vTop = cHP1[(t+1)%pCnt] - cHP[i]
            curr = vBottom.crossProduct(vTop)
            if(curr > last):
                last = curr
            else:
                break
            t = (t+1) % pCnt

        # 右侧r
        vRight = cHP1[r] - cHP[i]
        last = vBottom.dotProduct(vRight)
        curr = 0.0
        while(1):
            vRight = cHP1[(r+1)%pCnt] - cHP[i]
            curr = vBottom.dotProduct(vRight)
            if(curr > last):
                last = curr
            else:
                break
            r = (r+1) % pCnt

        # 左侧l
        if(i == 0):
            l = t
        vLeft = cHP1[l] - cHP[i]
        last = vBottom.dotProduct(vLeft)
        curr = 0.0
        while(1):
            vRight = cHP1[(l+1)%pCnt] - cHP[i]
            curr = vBottom.dotProduct(vRight)
            if(curr < last):
                last = curr
            else:
                break
            l = (l+1) % pCnt
        # 计算高和宽（不是最小外接矩形的高和宽，w*h数值上等于最小外接矩形面积）
        h = vBottom.crossProduct(cHP1[t]-cHP[i]) / vBottom.dotProduct(vBottom)
        w = vBottom.dotProduct(cHP1[r]-cHP[i]) - vBottom.dotProduct(cHP1[l]-cHP[i])
        tmpArea = w * h
        if(i == 0 or tmpArea < minArea):
            minArea = tmpArea
            minI = i
            minT = t
            minR = r
            minL = l
            minH = h
            minW = w
    # 由5点求出最小外接矩形参数
    p = getRectInfo(cHP[minI],cHP[minI+1],cHP1[minT],cHP1[minL],cHP1[minR])
    tmpW = math.sqrt(math.pow(p[0].x - p[1].x, 2) + math.pow(p[0].y - p[1].y, 2))
    tmpH = math.sqrt(math.pow(p[0].x - p[3].x, 2) + math.pow(p[0].y - p[3].y, 2))
    # 求最小外接矩形的短边与长边
    maxBorder = max(tmpW, tmpH)
    minBorder = min(tmpW, tmpH)
    rect = Rect2(p,minBorder,maxBorder,minArea)

    return rect

def getMER(src):
    # 根据标记获取边缘
    borderImg = getBinaryImageBorder(src,255)
    # 根据标记获取边缘点
    b = getBorderInfo(borderImg,255)
    # 获取凸包点集
    ch = getConvexHull(b)
    # 获取最小外接矩形
    minRect = getMinRectByRotatingCalipers(ch)

    return minRect


input = cv.imread("F://Test_Img//MBR.bmp",cv.IMREAD_GRAYSCALE)
cv.imshow("input",input)
minRect = getMER(input)
# 画出最小外接矩形
output = cv.cvtColor(input,cv.COLOR_GRAY2RGB)
cv.line(output,(round(minRect.p[0].y),round(minRect.p[0].x)),(round(minRect.p[1].y),round(minRect.p[1].x)),(0,255,0),1)
cv.line(output,(round(minRect.p[1].y),round(minRect.p[1].x)),(round(minRect.p[2].y),round(minRect.p[2].x)),(0,255,0),1)
cv.line(output,(round(minRect.p[2].y),round(minRect.p[2].x)),(round(minRect.p[3].y),round(minRect.p[3].x)),(0,255,0),1)
cv.line(output,(round(minRect.p[3].y),round(minRect.p[3].x)),(round(minRect.p[0].y),round(minRect.p[0].x)),(0,255,0),1)
cv.imshow("Min Rect",output)
cv.waitKey(0)
