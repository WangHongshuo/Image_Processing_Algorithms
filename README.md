# Image Algorithms
- 借助opencv的Mat的数据结构来读取图像，实现一些常见的常见的图像处理算法。

- ---> 程序没有进行单元测试，仅把公式简单的实现 <---
- ------------> 临界条件可能报错，仅供参考 <------------

## Image_Canny (C++): 
- [Canny边缘检测](./Image_Canny/Image_Canny/)： [原理参考链接](https://blog.csdn.net/jia20003/article/details/41173767)

## Image_Resize (C++):
- [图像缩放](./Image_Resize/Image_Resize/)
- nearest（最邻近插值）：[原理参考链接](https://www.cnblogs.com/korbin/p/5612427.html)
- linear（线性插值）：[原理参考链接](https://www.cnblogs.com/korbin/p/5612427.html)
- bicubic（双三次插值）：[原理参考链接](https://blog.csdn.net/u010979495/article/details/78428898)

## Image_Rotate(Python):
- [图像旋转](./Image_Rotate/Image_Rotate/)：[原理参考链接](https://blog.csdn.net/linshanxian/article/details/68944748)

## Image_Histogram_Equalization(Python):
- [直方图均衡化](./Image_Histogram_Equalization/Image_Histogram_Equalization)：[原理参考链接](https://www.cnblogs.com/tianyalu/p/5687782.html)

## Image_MER(Python):
- [求目标最小外接矩形](./Image_MER/Image_MER/)
- 求凸包（Graham扫描法）：[原理参考链接](https://www.cnblogs.com/Booble/archive/2011/03/10/1980089.html)
- 旋转卡壳法：[原理参考链接](https://blog.csdn.net/hanchengxi/article/details/8639476)

## Image_BWLable(Python):
- [二值连通区域标记](./Image_BWLable/Image_BWLable/)
- Two-Pass法：[原理参考链接](https://blog.csdn.net/hemeinvyiqiluoben/article/details/39854315)

## Image_DFT_IDFT(C++):
- [二维离散傅立叶变换（原始公式，速度非常慢）](./Image_DFT_IDFT/Image_DFT_IDFT/)

## Image_FFT_IFFT(C++):
- [二维离散快速傅立叶变换](./Image_FFT_IFFT/Image_FFT_IFFT/)：[原理参考链接](https://www.cnblogs.com/Lyush/articles/3219196.html)

## Image_Hough(C++):
- [霍夫变换](./Image_Hough/Image_Hough/)：[原理参考链接](https://www.cnblogs.com/yunlambert/p/7487582.html)

## Image_Radon(C++):
- [Radon变换](./Image_Radon/Image_Radon/)：[原理参考链接](https://blog.csdn.net/xiaoshen0121/article/details/79437957)

## Image_White_Balance(C++):
- [图像白平衡](./Image_White_Balance/Image_White_Balance/)
- 灰度世界算法：[原理参考连接](http://www.cnblogs.com/Imageshop/archive/2013/04/20/3032062.html)
- 完美反射镜法：[原理参考链接](http://www.cnblogs.com/Imageshop/archive/2013/04/20/3032062.html)