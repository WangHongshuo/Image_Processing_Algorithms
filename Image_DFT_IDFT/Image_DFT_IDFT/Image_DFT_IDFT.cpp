#include "pch.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;

/** @fn                                     将振幅谱低频部分平移到图像中心
 *  @param  src                             输入图像
 *  @return                                 void
 */
void fftShift(Mat &src)
{
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    Mat temp;
    Mat q0(src, Rect(0, 0, cx, cy));
    Mat q1(src, Rect(cx, 0, cx, cy));
    Mat q2(src, Rect(0, cy, cx, cy));
    Mat q3(src, Rect(cx, cy, cx, cy));
    // 交换左上右下、右上左下
    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);
}

/** @fn                                     恢复平移
 *  @param  src                             输入图像
 *  @return                                 void
 */
void ifftShift(Mat& src)
{
    fftShift(src);
}

/** @fn                                     原始二维离散傅立叶变换
 *  @param  src                             输入图像
 *  @param  re                              离散傅立叶变换后的实部
 *  @param  im                              离散傅立叶变换后的虚部
 *  @param  am                              log 能量谱
 *  @return                                 void
 */
void originalDFT(const Mat& src, Mat& re, Mat& im, Mat& am)
{
    re = Mat(src.size(), CV_64FC1);
    im = Mat(src.size(), CV_64FC1);
    int rows = src.rows, cols = src.cols;
    double tmpRe, tmpIm, t;
    // 二维离散傅立叶变换公式，速度非常慢，建议测试100*100以下图像
    for (int u = 0; u < rows; ++u)
    {
        for (int v = 0; v < cols; ++v)
        {
            tmpRe = 0;
            tmpIm = 0;
            for (int x = 0; x < rows; ++x)
            {
                for (int y = 0; y < cols; ++y)
                {
                    // 欧拉公式
                    t = -2 * CV_PI*(double(u * x) / double(rows) + double(v * y) / double(cols));
                    tmpRe += src.at<uchar>(x, y)*cos(t);
                    tmpIm += src.at<uchar>(x, y)*sin(t);
                }
            }
            re.at<double>(u, v) = tmpRe;
            im.at<double>(u, v) = tmpIm;
        }
    }
    // 生成log能量谱
    am = Mat(src.size(), CV_64FC1);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            am.at<double>(i, j) = pow(re.at<double>(i, j), 2) + pow(im.at<double>(i, j), 2) + 1;
            am.at<double>(i, j) = log(am.at<double>(i, j));
        }
    }
    normalize(am, am, 0, 255, NORM_MINMAX);
    am.convertTo(am, CV_8UC1);
    fftShift(am);
}

/*  @fn                                     原始二维离散傅立叶逆变换
 *  @param  re                              二维离散傅立叶变换后的实部
 *  @param  im                              二维离散傅立叶变换后的虚部
 *  @param  dst                             逆变换后的图像
 *  @return                                 void
 */
void originalIDFT(const Mat& re, const Mat& im, Mat& dst)
{
    dst = Mat(re.size(), CV_64FC1);
    Mat dstRe = Mat(re.size(), CV_64FC1);
    Mat dstIm = Mat(re.size(), CV_64FC1);
    int rows = re.rows, cols = re.cols;
    double MN = 1.0 / (rows * cols);
    double tmpRe, tmpIm, tRe, tIm, t;
    // 二维离散傅立叶逆变换公式，速度非常慢，建议测试100*100以下图像
    for (int x = 0; x < rows; ++x)
    {
        for (int y = 0; y < cols; ++y)
        {
            tmpRe = 0;
            tmpIm = 0;
            for (int u = 0; u < rows; ++u)
            {
                for (int v = 0; v < cols; ++v)
                {
                    // 欧拉公式
                    t = 2 * CV_PI*(double(u * x) / double(rows) + double(v * y) / double(cols));
                    tRe = cos(t);
                    tIm = sin(t);
                    tmpRe += re.at<double>(u, v)*tRe - im.at<double>(u, v)*tIm;
                    tmpIm += im.at<double>(u, v)*tRe + re.at<double>(u, v)*tIm;
                }
            }
            dstRe.at<double>(x, y) = MN * tmpRe;
            dstIm.at<double>(x, y) = MN * tmpIm;
        }
    }
    // 取模
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            dst.at<double>(i, j) = sqrt(pow(dstRe.at<double>(i, j), 2) + pow(dstIm.at<double>(i, j), 2));
        }
    }
    dst.convertTo(dst, CV_8UC1);
}

int main()
{
    Mat input = imread("F://MBR.bmp", IMREAD_GRAYSCALE);
    Mat re, im, am, output;
    originalDFT(input, re, im, am);
    originalIDFT(re, im, output);
    imshow("input", input);
    imshow("log amplitude", am);
    imshow("IDFT img", output);
    waitKey(0);
    return 0;
}