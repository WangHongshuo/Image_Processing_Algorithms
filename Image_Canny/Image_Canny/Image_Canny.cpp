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

float gaussFunc(int x, int y, float theta)
{
	// 二维高斯函数
	theta *= theta;
	float pi = 3.1415926;
	float p1 = -(x * x + y * y) / 2.0 / theta;
	float p2 = 1 / (2 * pi * theta);
	return p2 * exp(p1);
}

template<typename T>
void convFunc(const Mat& src, const Mat& k, Mat& dst, T &dateTpye)
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
			for (int x = 0; x < k.rows; x++)
			{
				xR = std::abs(i + (x - c));
				for (int y = 0; y < k.cols; y++)
				{
					// 边缘采用镜像复制
					yR = std::abs(j + (y - c));
					if (xR >= src.rows)
						xR = 2 * src.rows - xR - 1;
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

Mat gaussFilter(const Mat& src, size_t filterSize, float theta)
{
	Mat dst(src.size(), src.type());
	if (filterSize % 2 == 0)
		filterSize++;
	// 生成卷积核
	Mat k(filterSize, filterSize, CV_32FC1);
	size_t r = filterSize / 2;
	for (size_t i = r; i < filterSize; i++)
		for (size_t j = r; j < filterSize; j++)
			k.at<float>(i, j) = gaussFunc(i - r, j - r, theta);

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
	uchar tpye = 1;
	convFunc(src, k, dst, tpye);
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

Mat cannyFunc(const Mat& src, float th1, float th2)
{
	Mat dst = src.clone();
	dst.convertTo(dst, CV_32FC1);
	Mat theta(src.size(), CV_32FC1); // float
	Mat fx(src.size(), CV_32FC1); // float
	Mat fy(src.size(), CV_32FC1); // float
	Mat kx = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat ky = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	float type = 1;
	// Gx, Gy
	convFunc(dst, kx, fx, type);
	convFunc(dst, ky, fy, type);
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
			if (cX1 >= 0 && cX1 < dst.rows && cY1 >= 0 && cY1 < dst.rows)
				tmp1 = dstTmp.at<float>(cX1, cY1);
			else
				tmp1 = -1;
			if (cX1 >= 0 && cX1 < dst.rows && cY1 >= 0 && cY1 < dst.rows)
				tmp2 = dstTmp.at<float>(cX2, cY2);
			else
				tmp2 = -1;
			if (tmp1 > dst.at<float>(i, j) || tmp2 > dst.at<float>(i,j))
				dst.at<float>(i, j) = 0;
		}
	}
	//  双阈值边缘连接
	bool flag1, flag2;
	for (int i = 1; i < dst.rows; i++)
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
					if (x == y == 0)
						continue;
					int cX = std::abs(i + x);
					int cY = std::abs(j + y);
					if (cX > dst.rows)
						cX = 2 * dst.rows - cX;
					if (cY > dst.cols)
						cY = 2 * dst.cols - cY;
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