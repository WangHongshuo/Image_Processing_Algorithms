#include "pch.h"
#include <iostream>
#include <vector>
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
    if (src.type() != CV_8UC3)
    {
        cout << "The image must be 8bit and 3 channels image." << endl;
        abort();
        return;
    }
    // clone()用来保证图像数据在内存连续
    dst = src.clone();
    // int64 == long long，如果图像过大注意溢出，可采用分行计算平均值方法避免
    int64 pixelNum = src.total();
    int64 pixelBGRSum[3] = { 0,0,0 };
    int64 i;
    for (i = 0; i < pixelNum*dst.channels(); i += 3)
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
        dst.data[i + 1] = std::min(int(double(dst.data[i + 1])*k[1]), 255);
        dst.data[i + 2] = std::min(int(double(dst.data[i + 2])*k[2]), 255);
    }
}

/** @fn                                     完美反射法
 *  @param  src                             输入RGB图像
 *  @param  ratio                           前ratio %的白色参考点阈值，范围(0,1)
 *  @param  dst                             白平衡后的图像
 *  @return                                 void
 */
void perfectReflector(const Mat& src, double ratio, Mat& dst)
{
    if (src.type() != CV_8UC3)
    {
        cout << "The image must be 8bit and 3 channels image." << endl;
        abort();
        return;
    }
    if (ratio < 0.00000001 || 1.0 - ratio < 0.00000001)
    {
        cout << "ratio > 0.0 && ratio < 1.0" << endl;
        abort();
        return;
    }
    // clone()用来保证图像数据在内存连续
    dst = src.clone();
    Mat sumSrc(src.size(), CV_16UC1);
    vector<int> hist(255 * 3 + 1);
    int64 pixelNum = src.total();
    int tmp;
    // 计算三通道的和，保存在sumSrc中，并统计hist方便求前ratio%的白色参考点阈值
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            tmp = 0;
            tmp += src.at<Vec3b>(i, j)[0];
            tmp += src.at<Vec3b>(i, j)[1];
            tmp += src.at<Vec3b>(i, j)[2];
            hist[tmp]++;
            sumSrc.at<short>(i, j) = short(tmp);
        }
    }
    // 求白色参考点阈值
    short threshold;
    int tPixelNum = (int)std::round(double(pixelNum)*ratio);
    tmp = 0;
    double maxVal;
    for (int i = 255 * 3; i >= 0; i--)
    {
        tmp += hist[i];
        if (tmp > tPixelNum)
        {
            threshold = i;
            break;
        }
    }
    // 求最大白色参考点
    for (int i = 255 * 3; i >= 0; i--)
    {
        if (hist[i] > 0)
        {
            maxVal = double(i) / 3.0;
            break;
        }
    }
    // 把sumSrc中大于阈值的点，对应的src上的点的BGR累加，求平均
    double avgB = 0, avgG = 0, avgR = 0;
    double cnt = double(tmp);
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (sumSrc.at<ushort>(i, j) >= threshold)
            {
                avgB += src.at<Vec3b>(i, j)[0];
                avgG += src.at<Vec3b>(i, j)[1];
                avgR += src.at<Vec3b>(i, j)[2];
            }
        }
    }
    avgB /= cnt;
    avgG /= cnt;
    avgR /= cnt;
    int nB, nG, nR;
    // 计算白平衡后的新BGR值，并映射到[0,255]
    for (int i = 0; i < pixelNum*src.channels(); i += 3)
    {
        nB = int(double(dst.data[i])*maxVal / avgB);
        nG = int(double(dst.data[i + 1])*maxVal / avgG);
        nR = int(double(dst.data[i + 2])*maxVal / avgR);
        nB = std::max(0, std::min(nB, 255));
        nG = std::max(0, std::min(nG, 255));
        nR = std::max(0, std::min(nR, 255));
        dst.data[i] = uchar(nB);
        dst.data[i + 1] = uchar(nG);
        dst.data[i + 2] = uchar(nR);
    }
}

int main()
{
    Mat input = imread("F://Test_Img//WB.jpg");
    Mat GW_output, PR_output;
    grayWorld(input, GW_output);
    perfectReflector(input, 0.1, PR_output);
    imshow("ori img", input);
    imshow("gray world res", GW_output);
    imshow("perfect reflector res", PR_output);
    waitKey(0);
    return 0;
}