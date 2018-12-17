import cv2 as cv
import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import copy
import datetime

#   @class                              二值区域信息
#   @member label                       区域的标记
#   @member size                        区域像素个数
#   @member topLeft                     区域外接矩形左上点
#   @member bottomRigh                  区域外接矩形右下点
class BwArea:
    def __init__(self,_label = -1,_size = 0):
        self.label = _label
        self.size = _size
        self.topLeft = [99999999,99999999]
        self.bottomRight = [-1,-1]

#   @fn                                 更新区域外接矩形两个角点
#   @param  _x                          输入x坐标
#   @param  _y                          输入y坐标
#   @return                             返回自己
    def updatePos(self,_x,_y):
        self.topLeft[0] = min(self.topLeft[0],_x)
        self.topLeft[1] = min(self.topLeft[1],_y)
        self.bottomRight[0] = max(self.bottomRight[0],_x)
        self.bottomRight[1] = max(self.bottomRight[1],_y)
        return self

#   @notice                             list index（list中下标） == root的Label为某集合的root
#   @class                              标记并查集
#   @member label                       存放class Label的list()
#   @member normLabel                   root被归一化后的结果
#   @member __uniqueRoot                root合集
#   @member __size                      并查集大小
#   @member __rootCnt                   root数量
class LabelUnionFind:

    def __init__(self):
        self.label = list()
        self.normLabel = list()
        self.__uniqueRoot = list()
        self.__size = 0
        self.__rootCnt = 0

    #   @fn                             返回LabelUnionFind.label的大小
    #   @return                         LabelUnionFind.label的大小
    def size(self):
        return self.__size

    #   @fn                             返回Root Label的数量
    #   @return                         Root Label的数量
    def rootCnt(self):
        return self.__rootCnt

    #   @fn                             返回LabelUnionFind中最后面的Label
    #   return                          LabelUnionFind中最后面的Label
    def backLabel(self):
        return self.__size - 1

    #   @fn                             在list末尾添加新Label
    #   @param  _parent                 新Label的parent
    #   @return                         void
    def addLabel(self, _parent):
        if(_parent < 0 or _parent > self.__size):
            print("Invalid parent.")
            raise
        self.label.append(self.label[_parent])
        self.__size += 1

    #   @fn                             在list末尾添加新Root Label
    #   @return                         void
    def addRootLabel(self):
        self.label.append(self.__size)
        self.__size += 1
        self.__rootCnt += 1

    #   @fn                             归一化Label的root，从0开始映射，存放到LabelUnionFind.normLabel中
    #   @return                         void
    def normlizeLabelRoot(self):
        self.__uniqueRoot.clear()
        self.normLabel = [0] * self.__size
        for i in range(0, len(self.label)):
            if(i == self.label[i]):
                self.__uniqueRoot.append(i)

        # 创建一个dict，将Label.root归一化从0开始
        mappedLabel = dict()
        for i in range(0, self.__rootCnt):
            mappedLabel[self.__uniqueRoot[i]] = i 
        # 映射
        for i in range(0, self.__size):
            self.normLabel[i] = mappedLabel[self.label[i]]


    #   @fn                             合并两个Label
    #   @param  _srcLabel1              源Label
    #   @param  _dstLabel2              目标Label
    #   @return                         void
    def uinonLabel(self, _srcLabel, _dstLabel):
        if(_srcLabel < 0 or _dstLabel < 0 or _srcLabel > self.__size or _dstLabel > self.__size):
            print("Invalid label.")
            raise
        # root相等则跳过
        if(self.label[_srcLabel] == self.label[_dstLabel]):
            return
        # 否则遍历更改所有与srcLabel有相同root的Label的root，这里可以建立root : children的dict来减少循环，空间换速度
        else:
            srcRoot = self.label[_srcLabel]
            dstRoot = self.label[_dstLabel]
            for i in range(0, self.__size):
                if(self.label[i] == srcRoot):
                    self.label[i] = dstRoot
            self.__rootCnt -= 1


#   @fn                                 two-pass法标记联通区域
#   @param  src                         输入二值图像(0 - 255)
#   @return                             连通区域信息
def twoPass(src, neighbor = 4):

    startTime = datetime.datetime.now()

    if(neighbor != 4 and neighbor != 8):
        print("Invalid neighbor parameter!")
        raise

    # 扩充上限以标记更多连通区域
    dst = np.zeros(src.shape,dtype = 'uint16')
    rows = src.shape[0]
    cols = src.shape[1]
    MAX_PIXEL_VAL = 255

    print("two-pass init time: ", datetime.datetime.now() - startTime)
    tmpTime = datetime.datetime.now()

    # two-pass
    # LabelUnionFind.label的下标为Label
    # LabelUnionFind.label[i]为该Label的root
    luf = LabelUnionFind()
    # Label从1开始，跳过0
    luf.addRootLabel()
    if(neighbor == 4):
        for i in range(0,rows):
            for j in range(0,cols):
                # 该点为目标时
                if(src[i][j] == MAX_PIXEL_VAL):
                    top = 0
                    left = 0
                    if(i-1 >= 0):
                        top = dst[i-1][j]
                    if(j-1 >= 0):
                        left = dst[i][j-1]
                    # 左和上都为无效像素值，赋新Label并加入LabelUnionFind中
                    if(top == 0 and left == 0):
                        luf.addRootLabel()
                        dst[i][j] = luf.backLabel()
                    # 左和上都为Label时
                    elif(top > 0 and left > 0):
                        # 如果Label相等，则该点赋左或上的值
                        if(top == left):
                            dst[i][j] = top
                        # 如果Label不相等，则该点赋两点最小的label，并合并Label
                        else:
                            minVal = min(top,left)
                            maxVal = max(top,left)
                            dst[i][j] = np.uint16(minVal)
                            luf.uinonLabel(maxVal, minVal)
                    # 如果左和上只有一个点有效，则该点赋有效点的Label
                    elif(top > 0 or left > 0):
                        dst[i][j] = max(top,left)
    else:
        tmpSet = set()
        for i in range(0, rows):
            for j in range(0, cols):
                # 当该点为目标时
                if(src[i][j] == MAX_PIXEL_VAL):
                    # 依次读取left, topLeft, top, topRight到set中，去零去重
                    tmpSet.clear()
                    if(j-1 >= 0):
                        tmpSet.add(dst[i][j-1])         # top
                        if(i-1 >= 0):
                            tmpSet.add(dst[i-1][j-1])   # topLeft
                    if(i-1 >= 0):
                        tmpSet.add(dst[i-1][j])         # top
                        if(j+1 < cols):
                            tmpSet.add(dst[i-1][j+1])   # topRight
                    tmpSet.discard(0)

                    # tmpSet大小为0，则周围四点全为0，开新Label
                    if(len(tmpSet) == 0):
                        luf.addRootLabel()
                        dst[i][j] = luf.backLabel()
                    # tmpSet大小为1，则该点赋值为存在的这一点
                    elif(len(tmpSet) == 1):
                        dst[i][j] = next(iter(tmpSet))
                    # tmpSet大于1，则从小到大排序，该点赋值最小的Label，并把其他Label与最小Label合并
                    else:
                        tmpList = list(tmpSet)
                        tmpList.sort()
                        dst[i][j] = tmpList[0]
                        for k in range(1, len(tmpList)):
                            luf.uinonLabel(tmpList[k], tmpList[0])

    luf.normlizeLabelRoot()
    print("two-pass get label time: ", datetime.datetime.now() - tmpTime)
    tmpTime = datetime.datetime.now()

    # 建立area list
    areas = list()
    for i in range(0, luf.rootCnt()):
        areas.append(BwArea(i))

    # 使单个连通区域内的Label一致化，同时更新areas list信息
    for i in range(0,rows):
        for j in range(0,cols):
            if(dst[i][j] > 0):
                dst[i][j] = luf.normLabel[dst[i][j]]
                areas[dst[i][j]].size += 1
                areas[dst[i][j]] = areas[dst[i][j]].updatePos(i,j)

    print("two-pass label dst img time: ", datetime.datetime.now() - tmpTime)
    tmpTime = datetime.datetime.now()

    # 连通区域标记不同颜色
    labeledImg = copy.deepcopy(src)
    labeledImg = cv.cvtColor(labeledImg,cv.COLOR_GRAY2BGR)
    for i in range(0,rows):
        for j in range(0,cols):
            labeledImg[i][j] = [(dst[i][j]*121)%255,(dst[i][j]*246)%255,(dst[i][j]*336)%255]
    # 连通区域标记
    for i in range(1,len(areas)):
        cv.putText(labeledImg,str(areas[i].label),
                   (areas[i].bottomRight[1],areas[i].bottomRight[0]),
                   cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)

    print("two-pass draw dst img time: ", datetime.datetime.now() - tmpTime)
    print("two-pass total time: ", datetime.datetime.now() - startTime)
    cv.imshow("labeledImg",labeledImg)
    return areas

input = cv.imread("F://Test_Img//BWLabel.bmp",cv.IMREAD_GRAYSCALE)
input = cv.threshold(input,20,255,cv.THRESH_BINARY)
input = input[1]
areas = twoPass(input, 8)
print("areas count: ", len(areas) - 1)
cv.imshow("input",input)
cv.waitKey(0)
