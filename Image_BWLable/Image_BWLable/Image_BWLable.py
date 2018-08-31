import cv2 as cv
import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import copy

#   @fn                                 two-pass法标记联通区域
#   @param  src                         输入二值图像(0 - 255)
#   @return                             标记后的图像
def twoPass(src):
    # 扩充上限以标记更多连通区域
    dst = np.zeros(src.shape,dtype = 'uint16')
    rows = src.shape[0]
    cols = src.shape[1]
    MAX_PIXEL_VAL = 255
    # two-pass
    label = np.uint16(1)
    # union-find，下标为label，uf[label]为映射的parent label
    uf = list([np.uint16(0)])
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
                # 左和上都为无效像素值，赋新label并加入union-find中
                if(top == 0 and left == 0):
                    dst[i][j] = label
                    uf.append(label)
                    label += 1
                # 左和上都为label时
                elif(top > 0 and left > 0):
                    # 如果label相等，则该点赋左或上的值
                    if(top == left):
                        dst[i][j] = top
                    # 如果label不相等，则该点赋两点最小的label，并更新union-find
                    else:
                        minVal = min(top,left)
                        maxVal = max(top,left)
                        dst[i][j] = np.uint16(minVal)
                        uf[maxVal] = np.uint16(minVal)
                # 如果左和上只有一个点有效，则该点赋有效点的label
                elif(top > 0 or left > 0):
                    dst[i][j] = max(top,left)

    # 更新union-find，使每一个label映射到相应的集合里
    for i in range(1,len(uf)):
        mark = uf[i]
        # 如果某label映射自己，则该label为root label，否则循环找该label映射parent label并直到找到root label
        while(uf[mark] != mark):
            mark = uf[mark]
        uf[i] = mark

    # 使单个连通区域内的label一致化
    for i in range(0,rows):
        for j in range(0,cols):
                dst[i][j] = uf[dst[i][j]]

    # 给连通区域标记颜色
    labeledImg = copy.deepcopy(src)
    labeledImg = cv.cvtColor(labeledImg,cv.COLOR_GRAY2BGR)
    for i in range(0,rows):
        for j in range(0,cols):
            labeledImg[i][j] = [(dst[i][j]*121)%255,(dst[i][j]*246)%255,(dst[i][j]*336)%255]
    return labeledImg
    
input = cv.imread("F://MBR.bmp",cv.IMREAD_GRAYSCALE)
input = cv.threshold(input,20,255,cv.THRESH_BINARY)
input = input[1]
res = twoPass(input)
cv.imshow("input",input)
cv.imshow("labeledImg",res)
cv.waitKey(0)
