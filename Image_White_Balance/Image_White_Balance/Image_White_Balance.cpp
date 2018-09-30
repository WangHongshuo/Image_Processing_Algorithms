#include "pch.h"
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

using namespace std;
using namespace cv;

/** @fn                                     灰度世界算法
 *  @param  src                             输入RGB图像
 *  @param  dst                             白平衡后的图像
 *  @return                                 void
 */
void grayWorld(const Mat& src, Mat& dst)
{
    if (src.channels() != 3)
    {
        cout << "The image must has 3 channels." << endl;
        system("pause");
        return;
    }
    // clone()用来保证图像数据在内存连续
    dst = src.clone();
    // int64 == long long，如果图像过大注意溢出，可采用分行计算平均值方法避免
    int64 pixelNum = src.total();
    int64 pixelBGRSum[3] = { 0,0,0 };
    int64 i;
    for (i = 0; i < pixelNum*dst.channels();i+=3)
    {
        pixelBGRSum[0] += dst.data[i];
        pixelBGRSum[1] += dst.data[i + 1];
        pixelBGRSum[2] += dst.data[i + 2];
    }
    // 每个通道像素平均值
    double avgBRG[3];
    avgBRG[0] = double(pixelBGRSum[0]) / double(pixelNum);
    avgBRG[1] = double(pixelBGRSum[1]) / double(pixelNum);
    avgBRG[2] = double(pixelBGRSum[2]) / double(pixelNum);
    double g = (avgBRG[0] + avgBRG[1] + avgBRG[2]) / 3;
    // 计算增益
    double k[3];
    k[0] = g / avgBRG[0];
    k[1] = g / avgBRG[1];
    k[2] = g / avgBRG[2];
    // 防止溢出，令大于255的像素点为255
    for (i = 0; i < pixelNum*dst.channels(); i += 3)
    {
        dst.data[i] = std::min(int(double(dst.data[i])*k[0]), 255);
        dst.data[i+1] = std::min(int(double(dst.data[i+1])*k[1]), 255);
        dst.data[i+2] = std::min(int(double(dst.data[i+2])*k[2]), 255);
    }
}

int main()
{
    Mat input = imread("F://Test_Img//lena.jpg");
    Mat output;
    grayWorld(input, output);
    imshow("ori img", input);
    imshow("output img", output);
    waitKey(0);
    return 0;
}
