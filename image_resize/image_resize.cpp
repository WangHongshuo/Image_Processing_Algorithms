/*		opencv坐标系(row, col)
 *		对应图象坐标系(y, x)
 */
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

Mat nearest(const Mat& src, int rows, int cols)
{
	Mat dst(rows, cols, src.type());
	// 倍率
	double xA = double(src.cols) / double(cols);
	double yA = double(src.rows) / double(rows);
	for (int i = 0; i < rows; ++i)
	{
		int sI = std::min((int)std::round(i*yA), src.rows - 1);
		int sJ;
		for (int j = 0; j < cols; j++)
		{
			sJ = std::min((int)std::round(j*xA), src.cols - 1);
			dst.at<uchar>(i, j) = src.at<uchar>(sI, sJ);
		}
	}
	return dst;
}

Mat linear(const Mat& src, int rows, int cols)
{
	Mat dst(rows, cols, src.type());
	// 倍率
	double xA = double(src.cols) / double(cols);
	double yA = double(src.rows) / double(rows);
	double cX, cY, u, v, tmpSum;
	int sX, sY, sX1, sY1;
	for (int x = 0; x < rows; x++)
	{
		cX = x * xA;
		sX = std::floor(cX);
		u = cX - sX;
		sX = std::min(sX, src.rows - 1);
		sX1 = std::min(sX + 1, src.rows - 1);
		for (int y = 0; y < cols; y++)
		{
			tmpSum = 0;
			cY = y * yA;
			sY = std::floor(cY);
			v = cY - sY;
			sY = std::min(sY, src.cols - 1);
			sY1 = std::min(sY + 1, src.cols - 1);
			// f(sX+u,sY+v) = (1-u)(1-v)f(sX,sY) + (1-u)vf(sX,sY+1) + u(1-v)f(sX+1,sY) + uvf(sX+1,sY+1)
			tmpSum += (1 - u)*(1 - v)*src.at<uchar>(sX, sY);
			tmpSum += (1 - u)*v*src.at<uchar>(sX, sY1);
			tmpSum += u * (1 - v)*src.at<uchar>(sX1, sY);
			tmpSum += u * v*src.at<uchar>(sX1, sY1);
			tmpSum = std::min(tmpSum, 255.0);
			dst.at<uchar>(x, y) = (uchar)tmpSum;
		}
	}
	return dst;
}

Mat addBorder(const Mat& src, int l, int r)
{
	Mat dst(src.rows + l + r, src.cols + l + r, src.type());
	Mat roi(dst, cv::Rect(l, l, src.cols, src.rows));
	src.copyTo(roi);
	int rowCnt, colCnt, cI, cJ, k;
	rowCnt = 0;
	k = 0;
	for (int i = l - 1; i >= 0; --i)
	{
		cI = rowCnt + l - k * src.rows;
		for (int j = l; j < dst.cols - r; ++j)
		{
			dst.at<uchar>(i, j) = dst.at<uchar>(cI, j);
		}
		++rowCnt;
		if (rowCnt >= src.rows)
		{
			rowCnt = 0;
			++k;
		}
	}
	colCnt = 0;
	k = 0;
	for (int j = dst.cols - r; j < dst.cols; ++j)
	{
		cJ = dst.cols - r - 1 - colCnt + k * src.cols;
		for (int i = 0; i < dst.rows - r; ++i)
		{
			dst.at<uchar>(i, j) = dst.at<uchar>(i, cJ);
		}
		++colCnt;
		if (colCnt >= src.cols)
		{
			colCnt = 0;
			++k;
		}
	}
	rowCnt = 0;
	k = 0;
	for (int i = dst.rows - r; i < dst.rows; ++i)
	{
		cI = dst.rows - 1 - r - rowCnt + k * src.rows;
		for (int j = l - 1; j < dst.cols; ++j)
		{
			dst.at<uchar>(i, j) = dst.at<uchar>(cI, j);
		}
		++rowCnt;
		if (rowCnt >= src.rows)
		{
			rowCnt = 0;
			++k;
		}
	}
	colCnt = 0;
	k = 0;
	for (int j = l - 1; j >= 0; --j)
	{
		cJ = l + colCnt - k * src.cols;
		for (int i = 0; i < dst.rows; ++i)
		{
			dst.at<uchar>(i, j) = dst.at<uchar>(i, cJ);
		}
		++colCnt;
		if (colCnt >= src.cols)
		{
			colCnt = 0;
			++k;
		}
	}
	return dst;
}

double getS(double x, double a = -0.5)
{
	x = std::abs(x);
	if (x <= 1.0)
		return 1 - (a + 3)*std::pow(x, 2) + (a + 2)*std::pow(x, 3);
	else
		return -4 * a + 8 * a*x - 5 * a*std::pow(x, 2) + a * std::pow(x, 3);
}

Mat bicubic(const Mat& src, int rows, int cols)
{
	Mat dst(rows, cols, src.type());
	// 扩充src边界，扩充策略镜像复制
	Mat tmpSrc = addBorder(src, 1, 2);
	// 倍率
	double xA = double(src.cols) / double(cols);
	double yA = double(src.rows) / double(rows);
	double sI, sJ, v, u, res;
	for (int i = 0; i < dst.rows; ++i)
	{
		sI = i * xA;
		v = sI - std::floor(sI);
		sI = std::floor(sI);
		for (int j = 0; j < dst.cols; ++j)
		{
			sJ = j * yA;
			u = sJ - std::floor(sJ);
			sJ = std::floor(sJ);
			res = 0;
			for (int tI = -1; tI <= 2; ++tI)
			{
				for (int tJ = -1; tJ <= 2; ++tJ)
				{
					res += tmpSrc.at<uchar>(int(sI + 1 + tI), int(sJ + 1 + tJ))*getS(tI - v)*getS(tJ - u);
				}
			}
			dst.at<uchar>(i, j) = uchar(res);
		}
	}
	return dst;
}

int main()
{
	Mat input = imread("H://lena.jpg");
	cvtColor(input, input, COLOR_RGB2GRAY);
	double a = 1.5;
	/*      @fn                             image resize
	 *      @param  src                     输入图像
	 *      @param  rows                    缩放后的行数(image height)
	 *      @param  cols                    缩放后的列数(image width)
	 *      @return                         缩放后的图像
	 */
	Mat resNearest = nearest(input, input.rows*a, input.cols*a);
	Mat resLinear = linear(input, input.rows*a, input.cols*a);
	Mat resBicubic = bicubic(input, input.rows*a, input.cols*a);
	imshow("Origin", input);
	imshow("Nearest", resNearest);
	imshow("Linear", resLinear);
	imshow("Bicubic", resBicubic);
	waitKey(0);
	return 0;
}