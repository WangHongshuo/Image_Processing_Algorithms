#include "pch.h"
#include <iostream>
#include <vector>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;
/** @notice                                 hf(i,j)中，rho = i - hf.cols / 2, theta = j - 90
 *  @fn                                     霍夫变换
 *  @param  src                             输入边缘二值图像
 *  @param  hf                              霍夫累加器矩阵，hf(i,j): i -> rho, j -> theta
 *  @param  MARK                            有效标记
 *  @return                                 void
 */
void houghTransformation(const Mat& src, Mat& hf, int MARK)
{
    // 霍夫累加器矩阵高度
    int aRows = ceil((src.cols + src.rows) * std::sqrt(2));
    hf = Mat::zeros(aRows, 181, CV_32SC1);
    // 预计算cos(theta)和sin(theta)值
    vector<double> cosVal(181);
    vector<double> sinVal(181);
    int angle = -90;
    double x;
    for (int i = 0; i < 181; i++)
    {
        x = double(angle) / double(180)*CV_PI;
        cosVal[i] = cos(x);
        sinVal[i] = sin(x);
        ++angle;
    }
    cosVal[0] = cosVal[180] = 0;
    // 霍夫变换
    int rho;
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (src.at<uchar>(i, j) == MARK)
            {
                for (int theta = -90; theta <= 90; theta++)
                {
                    rho = std::round(i*cosVal[theta + 90] + j * sinVal[theta + 90]) + (aRows) / 2;
                    hf.at<int>(rho, theta + 90) += 1;
                }
            }
        }
    }
}

/** @fn                                     根据霍夫累加器画线
 *  @param  src                             输入图像
 *  @param  dst                             输出图像
 *  @param  hf                              霍夫累加器矩阵
 *  @param  minLin                          线长阈值，大于该阈值的线会被标出
 *  @param  color                           线的颜色
 *  @return                                 void
 */
void drawLines(const Mat& src, Mat& dst, const Mat& hf, const int minLen, Scalar color)
{
    vector<pair<int, vector<int>>> points;
    vector<int> v(2);
    int val, rhoBias = hf.rows / 2;
    dst = src.clone();
    cvtColor(dst, dst, CV_GRAY2BGR);
    // 寻找大于minLin长度的线
    for (int i = 0; i < hf.rows; i++)
    {
        for (int j = 0; j < hf.cols; j++)
        {
            val = hf.at<int>(i, j);
            if (val >= minLen)
            {
                v = { i - rhoBias,j };
                points.push_back(make_pair(val, v));
            }
        }
    }

    vector<double> cosVal(181);
    vector<double> sinVal(181);
    int angle = -90;
    double x;
    for (int i = 0; i < 181; i++)
    {
        x = double(angle) / double(180)*CV_PI;
        cosVal[i] = cos(x);
        sinVal[i] = sin(x);
        ++angle;
    }
    cosVal[0] = cosVal[180] = 0;
    // 画线
    int x0, y0, x1, y1, rho, theta;
    double k, b;
    for (auto &p : points)
    {
        rho = (p.second)[0];
        theta = (p.second)[1];
        if (sinVal[theta] == 0)
        {
            x0 = x1 = abs(rho);
            y0 = 0;
            y1 = dst.cols;
        }
        else
        {
            k = -(cosVal[theta] / sinVal[theta]);
            b = double(rho) / sinVal[theta];
            x0 = 0;
            y0 = floor(k * x0 + b);
            x1 = dst.rows;
            y1 = floor(k * x1 + b);
        }
        line(dst, Point(y0, x0), Point(y1, x1), color);
    }
}

int main()
{
    Mat input = imread("F://Test_Img//Hough.bmp", IMREAD_GRAYSCALE);
    threshold(input, input, 127, 255, THRESH_BINARY);
    Mat hf, output;
    houghTransformation(input, hf, 255);
    drawLines(input, output, hf, 50, Scalar(0, 255, 0));
    imshow("input image", input);
    imshow("hough image", hf);
    imshow("lines", output);
    waitKey(0);
    return 0;
}