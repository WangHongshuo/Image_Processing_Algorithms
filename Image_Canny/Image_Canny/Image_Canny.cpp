/*		opencv坐标系(row, col)
 *		对应图象坐标系(y, x)
 */
#include "pch.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

float gaussFunc(int x, int y, float sigma)
{
	// 二维高斯函数
	sigma *= sigma;
	float pi = 3.1415926;
	float p1 = -(x * x + y * y) / 2.0 / sigma;
	float p2 = 1 / (2 * pi * sigma);
	return p2 * exp(p1);
}

/** @fn                                     卷积，需传入数据类型convFunc<int>(src,k,dst)
 *  @param  <T>                             处理的数据类型
 *  @param  src                             输入矩阵
 *  @param  k                               卷积核
 *  @param  dst                             输出矩阵
 *  @return                                 void
 */
template<typename T>
void convFunc(const Mat& src, const Mat& k, Mat& dst)
{
	float tmp;
	int c = k.rows / 2;
    T res;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			tmp = 0;
			int xR, yR;
            // 边缘采用镜像复制
			for (int x = 0; x < k.rows; x++)
			{
				xR = std::abs(i + (x - c));
                if (xR >= src.rows)
                    xR = 2 * src.rows - xR - 1;
				for (int y = 0; y < k.cols; y++)
				{
					yR = std::abs(j + (y - c));
					if (yR >= src.cols)
						yR = 2 * src.cols - yR - 1;
					tmp += src.at<T>(xR, yR)*k.at<float>(x, y);
				}
			}
			res = std::floor(tmp);
			dst.at<T>(i, j) = res;
		}
	}
}
/** @fn                                     高斯滤波
 *  @param  src                             输入图像
 *  @param  filterSize                      滤波器大小
 *  @param  sigma                           高斯函数sigma
 *  @return                                 滤波后的图像
 */
Mat gaussFilter(const Mat& src, size_t filterSize, float sigma)
{
	Mat dst(src.size(), src.type());
	if (filterSize % 2 == 0)
		filterSize++;
	// 生成卷积核
	Mat k(filterSize, filterSize, CV_32FC1);
	size_t r = filterSize / 2;
	for (size_t i = r; i < filterSize; i++)
		for (size_t j = r; j < filterSize; j++)
			k.at<float>(i, j) = gaussFunc(i - r, j - r, sigma);

	for (size_t i = 0; i < r + 1; i++)
		for (size_t j = 0; j < r + 1; j++)
			k.at<float>(i,j) = k.at<float>(filterSize - 1 - i, filterSize - 1 - j);

	for (size_t i = 0; i < r; i++)
		for (size_t j = r + 1; j < filterSize; j++)
			k.at<float>(i, j) = k.at<float>(filterSize - 1 - i, j);

	for (size_t i = r + 1; i < filterSize; i++)
		for (size_t j = 0; j < r; j++)
			k.at<float>(i, j) = k.at<float>(i, filterSize - 1 - j);
	Scalar sum = cv::sum(k);
	k /= (float)sum[0];
	convFunc<uchar>(src, k, dst);
	return dst;
}

void atanFunc(const Mat& src, Mat& dst)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			dst.at<float>(i, j) = std::atan(src.at<float>(i, j));
}

void getCheckDirction(const Mat& src, int& dX, int& dY, int x, int y)
{
	float val = src.at<float>(x, y);
	if (val < 22.5)
	{
		dX = 0; dY = 1;
	}
	else if (val < 67.5)
	{
		dX = -1; dY = 1;
	}
	else if (val < 112.5)
	{
		dX = -1; dY = 0;
	}
	else if (val < 157.5)
	{
		dX = -1; dY = -1;
	}
	else
	{
		dX = 0; dY = -1;
	}
}
/** @fn                                     canny
 *  @param  src                             输入图像
 *  @param  th1                             双阈值边缘连接下限
 *  @param  th2                             双阈值边缘连接上限
 *  @return                                 边缘图像
 */
Mat cannyFunc(const Mat& src, float th1, float th2)
{
	Mat dst = src.clone();
	dst.convertTo(dst, CV_32FC1);
	Mat theta(src.size(), CV_32FC1); // float
	Mat fx(src.size(), CV_32FC1); // float
	Mat fy(src.size(), CV_32FC1); // float
	Mat kx = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat ky = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	// Gx, Gy
	convFunc<float>(dst, kx, fx);
	convFunc<float>(dst, ky, fy);
	cv::magnitude(fx, fy, dst);
	Mat dstTmp = dst.clone();
	// theta
	theta = fy / fx;
	atanFunc(theta, theta);
	float pi = 3.1415926;
	theta = (theta + pi / 2) / pi * 180;
	// 非最大信号压制算法
	int dX, dY, cX1, cY1, cX2, cY2;
	float tmp1, tmp2;
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			getCheckDirction(theta, dX, dY, i, j);
			cX1 = i + dX; cY1 = j + dY;
			cX2 = i - dX; cY2 = j - dY;
            // 可优先处理边缘，减少if-else
			if (cX1 >= 0 && cX1 < dst.rows && cY1 >= 0 && cY1 < dst.cols)
				tmp1 = dstTmp.at<float>(cX1, cY1);
			else
				tmp1 = -1.0;
			if (cX2 >= 0 && cX2 < dst.rows && cY2 >= 0 && cY2 < dst.cols)
				tmp2 = dstTmp.at<float>(cX2, cY2);
			else
				tmp2 = -1.0;
			if (tmp1 > dst.at<float>(i, j) || tmp2 > dst.at<float>(i,j))
				dst.at<float>(i, j) = 0;
		}
	}
	//  双阈值边缘连接
	bool flag1, flag2;
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			flag1 = true;
			flag2 = false;
			tmp1 = dstTmp.at<float>(i, j);
			if (tmp1 > th2)
				continue;
			if (tmp1 < th1)
			{
				dst.at<float>(i, j) = 0;
				continue;
			}
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					if (x == 0 && y == 0)
						continue;
                    // 可优先处理边缘，减少if
					int cX = std::abs(i + x);
					int cY = std::abs(j + y);
					if (cX >= dst.rows)
						cX = 2 * dst.rows - 1 - cX;
					if (cY >= dst.cols)
						cY = 2 * dst.cols - 1 - cY;
					if (dstTmp.at<float>(cX, cY) < th1)
						flag1 = false;
					if (dstTmp.at<float>(cX, cY) > th2)
						flag2 = true;
				}
			}
			if (!flag1 || !flag2)
				dst.at<float>(i, j) = 0;
		}
	}

	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<float>(i, j) > 0)
				dst.at<float>(i, j) = 255;
		}
	}
	dst.convertTo(dst, CV_8UC1);
	return dst;
}

int main()
{
	Mat input = imread("F://lena.jpg");
	cvtColor(input, input, CV_RGB2GRAY);
	Mat gaussRes = gaussFilter(input, 15, 1.5);
	Mat cannyRes = cannyFunc(gaussRes, 30, 60);
	imshow("src", input);
	imshow("gauss image", gaussRes);
	imshow("canny image", cannyRes);
	waitKey(0);
	return 0;
}