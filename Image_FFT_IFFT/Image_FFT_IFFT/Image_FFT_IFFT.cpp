#include "pch.h"
#include <iostream>
#include <vector>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/** @class                                  复数
 *  @member re                              实部
 *  @member im                              虚部
 */
template<class T>
class ComplexNum
{
public:
    T re;
    T im;
    ComplexNum() {}
    ComplexNum(T _re, T _im) { re = _re; im = _im; }
    ComplexNum(T _eIm) { this->euler(_eIm); }
    ~ComplexNum() {}
    void euler(T& eIm) { re = cos(eIm); im = sin(eIm); }
    ComplexNum operator+(const ComplexNum& cn)
    {
        ComplexNum<T> _t;
        _t.re = this->re + cn.re;
        _t.im = this->im + cn.im;
        return _t;
    }
    ComplexNum& operator+=(const ComplexNum& cn)
    {
        this->re += cn.re;
        this->im += cn.im;
        return *this;
    }
    ComplexNum operator*(const ComplexNum& cn)
    {
        ComplexNum<T> _t;
        _t.re = this->re * cn.re - this->im * cn.im;
        _t.im = this->re * cn.im + this->im * cn.re;
        return _t;
    }
    ComplexNum operator*(const T& val)
    {
        ComplexNum<T> _t;
        _t.re = this->re * val;
        _t.im = this->im * val;
        return _t;
    }
    ComplexNum& operator=(const T& val)
    {
        this->re = val;
        this->im = val;
        return *this;
    }
};

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

/** @notice                                 图像rows和cols不为偶数时，补零会影响结果 
 *  @fn                                     原始二维离散快速傅立叶变换
 *  @param  src                             输入图像
 *  @param  re                              离散傅立叶变换后的实部
 *  @param  im                              离散傅立叶变换后的虚部
 *  @param  am                              log 能量谱
 *  @return                                 void
 */
void originalFFT(const Mat& src, Mat& re, Mat& im, Mat& am)
{
    // 补成偶数，对结果有一定影响
    int rows = src.rows, cols = src.cols;
    if (rows % 2 == 1)
        rows += 1;
    if (cols % 2 == 1)
        cols += 1;
    Mat tmpSrc = Mat::zeros(rows, cols, src.type());
    Mat tmpSrcRoi = tmpSrc(Rect(0, 0, src.cols, src.rows));
    re = Mat::zeros(rows, cols, CV_64FC1);
    im = Mat::zeros(rows, cols, CV_64FC1);
    am = Mat(src.size(), CV_64FC1);
    Mat re2 = Mat(rows, cols, CV_64FC1);
    Mat im2 = Mat(rows, cols, CV_64FC1);
    src.copyTo(tmpSrcRoi);
    // 先按行进行一维FFT，后按列进行一维FFT，得到二维FFT
    double p = -2 * CV_PI, eIm1, eIm2;
    int hRows = rows / 2, hCols = cols / 2;
    ComplexNum<double> W1, W2, tmpSum, tmpEven, tmpOdd, tmpCN;
    // 行FFT
    for (int r = 0; r < rows; ++r)
    {
        for (int k = 0; k < cols; ++k)
        {
            tmpSum = 0;
            tmpEven = 0;
            tmpOdd = 0;
            eIm2 = p * k / cols;
            W2.euler(eIm2);
            for (int n = 0; n < cols; n += 2)
            {
                eIm1 = p * k * (n / 2) / hRows;
                W1.euler(eIm1);
                tmpEven += W1 * double(tmpSrc.at<uchar>(r,n));
                tmpOdd += W1 * double(tmpSrc.at<uchar>(r, n + 1));
            }
            tmpOdd = W2 * tmpOdd;
            tmpSum = tmpEven + tmpOdd;
            re.at<double>(r, k) = tmpSum.re;
            im.at<double>(r, k) = tmpSum.im;
        }
    }
    // 列FFT
    for (int l = 0; l < cols; ++l)
    {
        for (int k = 0; k < rows; ++k)
        {
            tmpSum = 0;
            tmpEven = 0;
            tmpOdd = 0;
            eIm2 = p * k / rows;
            W2.euler(eIm2);
            for (int n = 0; n < rows; n += 2)
            {
                eIm1 = p * k * (n / 2) / hCols;
                W1.euler(eIm1);
                tmpCN.re = re.at<double>(n, l);
                tmpCN.im = im.at<double>(n, l);
                tmpEven += W1 * tmpCN;
                tmpCN.re = re.at<double>(n + 1, l);
                tmpCN.im = im.at<double>(n + 1, l);
                tmpOdd += W1 * tmpCN;
            }
            tmpOdd = W2 * tmpOdd;
            tmpSum = tmpEven + tmpOdd;
            re2.at<double>(k, l) = tmpSum.re;
            im2.at<double>(k, l) = tmpSum.im;
        }
    }
    re = re2(Rect(0, 0, src.cols, src.rows));
    im = im2(Rect(0, 0, src.cols, src.rows));
    // 获得log能量谱
    for (int i = 0; i < am.rows; ++i)
    {
        for (int j = 0; j < am.cols; ++j)
        {
            am.at<double>(i, j) = pow(re.at<double>(i, j), 2) + pow(im.at<double>(i, j), 2) + 1;
            am.at<double>(i, j) = log(am.at<double>(i, j));

        }
    }
    normalize(am, am, 0, 255, NORM_MINMAX);
    am.convertTo(am, CV_8UC1);
    fftShift(am);
}

void originalIFFT()
{

}

int main()
{
    clock_t startTime, endTime;
    Mat input = imread("F://MBR.bmp", IMREAD_GRAYSCALE);
    Mat re, im, am;
    startTime = clock();
    originalFFT(input, re, im, am);
    endTime = clock();
    imshow("input image", input);
    imshow("amplitude image", am);
    cout << "Cost Time: " << (endTime - startTime) * 1000 / CLOCKS_PER_SEC << " MS" << endl;
    waitKey(0);
    return 0;
}
