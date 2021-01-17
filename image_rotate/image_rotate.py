import cv2 as cv
import numpy as np
import math

def rotateKernel(x,y,cosine,sine):
    x1 = x * cosine - y * sine
    y1 = x * sine + y * cosine
    return (x1, y1)

def iRotateKernel(x,y,cosine,sine):
    x1 = x * cosine + y * sine
    y1 = -x * sine + y * cosine
    return (x1, y1)

# 线性插值
def linear(x,y,src):
    srcX = math.floor(x)
    srcY = math.floor(y)
    if(srcX >= 0 and srcX < src.shape[1] and srcY >= 0 and srcY < src.shape[0]):
        u = x - srcX
        v = y - srcY
        srcX1 = min(srcX + 1,src.shape[1] - 1)
        srcY1 = min(srcY + 1,src.shape[0] - 1)
        res = 0
        # f(sX+u,sY+v) = (1-u)(1-v)f(sX,sY) + (1-u)vf(sX,sY+1) + u(1-v)f(sX+1,sY) + uvf(sX+1,sY+1)
        res = res + (1 - u)*(1-v)*src[srcY][srcX]
        res = res + (1 - u)*v*src[srcY1][srcX]
        res = res + u*(1 - v)*src[srcY][srcX1]
        res = res + u*v*src[srcY1][srcX1]
        for i in range(0,len(res)):
            res[i] = min(res[i],255)
        return res
    else:
        return (-1,)


def rotateFunc(image,center,angle,isExpand,method):
    ## opencv坐标系为(row, col)，对应图像坐标系(y, x)
    ## 旋转公式坐标系为(x, y)
    theta = -angle / 180 * math.pi
    cosine = math.cos(theta)
    sine = math.sin(theta)
    # 旋转中心(a, b)，原点平移到(a, b)
    a = center[1] # x - col
    b = center[0] # y - row
    srcRow = image.shape[0] # row - height - y
    srcCol = image.shape[1] # col - width  - x
    # 左上点
    x1 = -a
    y1 = b
    # 右上点
    x2 = srcCol - 1 - a
    y2 = b
    # 右下点
    x3 = srcCol -1 - a
    y3 = b - srcRow + 1
    # 左下点
    x4 = -a
    y4 = b - srcRow + 1
    ## 计算以(a, b)为坐标原点旋转后的角点并得出旋转后图像尺寸
    (x1, y1) = rotateKernel(x1,y1,cosine,sine)
    (x2, y2) = rotateKernel(x2,y2,cosine,sine)
    (x3, y3) = rotateKernel(x3,y3,cosine,sine)
    (x4, y4) = rotateKernel(x4,y4,cosine,sine)
    if (isExpand == 1):
        dstRow = round(max(abs(y1 - y3),abs(y2 - y4))) # row - height - y
        dstCol = round(max(abs(x1 - x3),abs(x2 - x4))) # col - width  - x
    else:
        dstRow = srcRow
        dstCol = srcCol
    dst = np.zeros((dstRow,dstCol,image.shape[2]),image.dtype)
    # 旋转后的中心
    if(isExpand == 1):
        c = dstCol // 2
        d = dstRow // 2
    else:
        c = a
        d = b
    f1 = -c * cosine + d * sine + a
    f2 = -c * sine - d * cosine + b
    for x in range(0,dstCol - 1):
        for y in range(0,dstRow - 1):
            (srcX ,srcY) = rotateKernel(x,y,cosine,sine)
            srcX = srcX + f1
            srcY = srcY + f2
            # 0 - nearest, 1 - linear
            if(method == 1):
                pixelVal = linear(srcX,srcY,image)
                srcX = math.floor(srcX)
                srcY = math.floor(srcY)
                if(not(len(pixelVal) == 1 and pixelVal[0] <= 0)):
                 dst[y][x] = pixelVal
            else:
                srcX = round(srcX)
                srcY = round(srcY)
                if(srcX >= 0 and srcX < srcCol and srcY >= 0 and srcY < srcRow):
                    dst[y][x] = image[srcY][srcX]
    return dst

input = cv.imread("H://lena.jpg")
#   @fn                     图像旋转
#   @param  image           输入图像
#   @param  center          旋转中心(row, col)
#   @param  angle           旋转角度，顺时针为正
#   @param  isExpand        0 - 保持和原图像一样大小，1 - 扩充图像
#   @param  method          0 - nearest， 1 - linear
#   @return                 旋转后的图像
res = rotateFunc(input,(input.shape[0] // 2, input.shape[1] // 2),45,1,1)
cv.imshow("input",input)
cv.imshow("rotated",res)
cv.waitKey(0)
