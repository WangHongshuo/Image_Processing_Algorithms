#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/** @notice                                 step过小会影响速度，已测试1.0和0.5，其他值可能会出现问题
 *  @fn                                     radon变换
 *  @param  src                             灰度图像
 *  @param  rt                              radon累加器矩阵
 *  @param  rtImg                           标准化和直方图均衡化后的radon累加器矩阵
 *  @param  markedImg                       标记最长直线的图像
 *  @param  step                            角度步长，默认1.0度
 *  @return                                 void
 */
void radonTransfromation(const Mat& src, Mat& rt, Mat& rtImg, Mat& markedImg, double step = 1.0)
{
    // 初始化radon累加器矩阵
    int aRow = (int)std::ceil((std::max(src.cols, src.rows)) * std::sqrt(2));
    int baisRho = aRow / 2;
    int thetaLen = (int)std::round(181.0 / step);
    rt = Mat::zeros(aRow, thetaLen, CV_32SC1);
    // 预先计算好cos(x)和sin(x)
    vector<double> cosVal(thetaLen);
    vector<double> sinVal(thetaLen);
    double angle = 0.0, x;
    for (int i = 0; i < thetaLen; i++)
    {
        x = double(angle) / 180.0 * CV_PI;
        cosVal[i] = cos(x);
        sinVal[i] = sin(x);
        angle += step;
    }
    sinVal[0] = sinVal[sinVal.size() - 1] = 0;
    // radon变换线积分
    // rho = x * cos(theta) + y * sin(theta)
    int rho;
    for (int theta = 0; theta < thetaLen; theta++)
    {
        for (int x = 0; x < src.rows; x++)
        {
            for (int y = 0; y < src.cols; y++)
            {
                rho = (int)std::round(x * cosVal[theta] + y * sinVal[theta]) + baisRho;
                if (rho >= 0 && rho < rt.rows)
                    rt.at<int>(rho, theta) += src.at<uchar>(x, y);
            }
        }
    }
    // 生成radon累加器的灰度图
    rtImg = rt.clone();
    normalize(rtImg, rtImg, 0, 255, NORM_MINMAX);
    rtImg.convertTo(rtImg, CV_8UC1);
    equalizeHist(rtImg, rtImg);
    // 寻找最长的线
    Point maxP;
    minMaxLoc(rt, nullptr, nullptr, nullptr, &maxP);
    markedImg = src.clone();
    cvtColor(markedImg, markedImg, COLOR_GRAY2BGR);
    int maxRho = maxP.y - baisRho;
    double mTheta = maxP.x * step;
    cout << "rho: " << maxRho << endl << "theta: " << mTheta << endl;
    int maxTheta = maxP.x;
    // 求直线
    double k, b;
    int x0, y0, x1, y1;
    if (maxTheta == 0 || maxTheta == sinVal.size() - 1)
    {
        x0 = x1 = abs(maxRho);
        y0 = 0;
        y1 = markedImg.cols;
    }
    else
    {
        k = -cosVal[maxTheta] / sinVal[maxTheta];
        b = double(maxRho) / sinVal[maxTheta];
        x0 = 0;
        y0 = (int)std::floor(k*x0 + b);
        x1 = markedImg.rows;
        y1 = (int)std::floor(k*x1 + b);
    }
    line(markedImg, Point(y0, x0), Point(y1, x1), Scalar(0,255,0));
}

int main()
{
    Mat input = imread("H://Test_Img//Radon.bmp", IMREAD_GRAYSCALE);
    Mat rt, normEHImg, markedImg;
    radonTransfromation(input, rt, normEHImg, markedImg, 1.0);
    imshow("input img", input);
    imshow("radon img", normEHImg);
    imshow("marked Img", markedImg);
    waitKey(0);
    return 0;
}