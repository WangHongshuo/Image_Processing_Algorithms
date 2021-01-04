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

/** @algorithm                              http://www.cnblogs.com/Imageshop/archive/2013/02/14/2911309.html
 *  @fn                                     BGR to YCbCr，Y、Cb、Cr分量取值范围为0~255
 *  @param  src                             输入三通道BGR图像
 *  @param  dst                             输出YCbCr图像
 *  @return                                 void
 */
void BGR2YCbCr(const Mat& src, Mat& dst)
{
    // RGB转YCbCr的系数(浮点类型）
    const float YCbCrYRF = 0.299F;
    const float YCbCrYGF = 0.587F;
    const float YCbCrYBF = 0.114F;
    const float YCbCrCbRF = -0.168736F;
    const float YCbCrCbGF = -0.331264F;
    const float YCbCrCbBF = 0.500000F;
    const float YCbCrCrRF = 0.500000F;
    const float YCbCrCrGF = -0.418688F;
    const float YCbCrCrBF = -0.081312F;
    const int Shift = 20;
    const int HalfShiftValue = 1 << (Shift - 1);
    // RGB转YCbCr的系数(整数类型）
    const int YCbCrYRI = (int)(YCbCrYRF * (1 << Shift) + 0.5);
    const int YCbCrYGI = (int)(YCbCrYGF * (1 << Shift) + 0.5);
    const int YCbCrYBI = (int)(YCbCrYBF * (1 << Shift) + 0.5);
    const int YCbCrCbRI = (int)(YCbCrCbRF * (1 << Shift) + 0.5);
    const int YCbCrCbGI = (int)(YCbCrCbGF * (1 << Shift) + 0.5);
    const int YCbCrCbBI = (int)(YCbCrCbBF * (1 << Shift) + 0.5);
    const int YCbCrCrRI = (int)(YCbCrCrRF * (1 << Shift) + 0.5);
    const int YCbCrCrGI = (int)(YCbCrCrGF * (1 << Shift) + 0.5);
    const int YCbCrCrBI = (int)(YCbCrCrBF * (1 << Shift) + 0.5);
    // 判断src和dst是否共用一个data
    Mat tDst;
    if (dst.datastart == src.datastart || dst.dataend == src.dataend || dst.data == nullptr)
        tDst = Mat(src.size(), src.type());
    else
        tDst = dst;
    int B, G, R;
    for (unsigned int i = 0; i < src.total() * 3; i += 3)
    {
        B = src.data[i];
        G = src.data[i + 1];
        R = src.data[i + 2];
        tDst.data[i] = (uchar)(((YCbCrYRI * R + YCbCrYGI * G + YCbCrYBI * B + HalfShiftValue) >> Shift));
        tDst.data[i + 1] = (uchar)(128 + ((YCbCrCbRI * R + YCbCrCbGI * G + YCbCrCbBI * B + HalfShiftValue) >> Shift));
        tDst.data[i + 2] = (uchar)(128 + ((YCbCrCrRI * R + YCbCrCrGI * G + YCbCrCrBI * B + HalfShiftValue) >> Shift));
    }
    dst = tDst;
}

/** @struct                                 图像Block
 *  @member img                             YCbCr图像
 *  @member Mb                              Cb分量映射到-128~127后的均值Mb
 *  @member Mr                              Cr分量映射到-128~127后的均值Mr
 *  @member Db                              Cb分量映射到-128~127后的平均绝对差值Db
 *  @member Dr                              Cr分量映射到-128~127后的平均绝对差值Dr
 */
struct Block
{
    Mat img;
    double Mb = 0.0;
    double Mr = 0.0;
    double Db = 0.0;
    double Dr = 0.0;
    Block(Mat &_img) { img = _img; };
};

/** @fn                                     计算YCbCr图像的Mb、Mr、Db和Dr
 *  @return                                 void
 */
void getMbMrDbDr(Block& src)
{
    double sumCb = 0.0, sumCr = 0.0;
    for (uint i = 0; i < src.img.total() * 3; i += 3)
    {
        sumCb += double(src.img.data[i + 1]);
        sumCr += double(src.img.data[i + 2]);
    }
    // Cb, Cr需要被映射到-128~127
    src.Mb = sumCb / double(src.img.total()) - 128.0;
    src.Mr = sumCr / double(src.img.total()) - 128.0;
    double tmpSumDb = 0.0, tmpSumDr = 0.0;
    for (uint i = 0; i < src.img.total() * 3; i += 3)
    {
        tmpSumDb += std::abs(double(src.img.data[i + 1]) - 128.0 - src.Mb);
        tmpSumDr += std::abs(double(src.img.data[i + 2]) - 128.0 - src.Mr);
    }
    src.Db = tmpSumDb / double(src.img.total());
    src.Dr = tmpSumDr / double(src.img.total());
}

/** @fn                                     Sign函数
 *  @param  input                           输入数值
 *  @return                                 Sign函数结果
 */
int mySignFunc(double input)
{
    if (input > 0.0000001)
        return 1;
    else if (input < -0.0000001)
        return -1;
    else
        return 0;
}

/** @notice                                 不保证完全还原论文中的算法，论文中有模糊的地方
 *  @algorithm                              https://files-cdn.cnblogs.com/files/Imageshop/ANovelAutomaticWhiteBalanceMethodforDigital.pdf
 *  @fn                                     自动阈值白平衡算法
 *  @param  src                             输入BGR图像
 *  @param  dst                             输出结果
 *  @param  blockWidth                      图像分块的块宽
 *  @param  blockHeight                     图像分块的块高
 *  @return                                 void
 */
void automaticWhiteBalance(const Mat& src, Mat& dst, size_t blockWidth, size_t blockHeight)
{
    if (src.type() != CV_8UC3)
    {
        cout << "The image must be 8bit and 3 channels image." << endl;
        abort();
        return;
    }
    if (blockWidth*blockWidth == 0 || blockWidth > (size_t)src.cols || blockHeight > (size_t)src.rows)
    {
        cout << "Invalid block size!" << endl;
        abort();
        return;
    }
    // 转化到YCbCr，每个通道的取值范围为0~255
    Mat YCbCrImg;
    dst = src.clone();
    BGR2YCbCr(dst, YCbCrImg);
    vector<Block> blocks;
    // 避免两个double相除后ceil()有误差(?)
    // x - col - width, y - row - height
    // 计算预分块数量
    int blockXNums = src.cols / blockWidth;
    if (src.cols % blockWidth != 0)
        blockXNums++;
    int blockYNums = src.rows / blockHeight;
    if (src.rows % blockHeight != 0)
        blockYNums++;
    blocks.reserve(blockXNums * blockYNums);
    // 分块有问题，因为图像不一定全都被分为大小相等的块，边缘的块大小与设置的块大小不同，可能对结果有影响
    int topLeftX = 0, topLeftY = 0, bottomRightX, bottomRightY, width, height;
    Mat tmpBlockImg;
    for (int i = 0; i < blockXNums; ++i)
    {
        bottomRightX = std::min(topLeftX + (int)blockWidth - 1, src.cols - 1);
        width = bottomRightX - topLeftX + 1;
        topLeftY = 0;
        for (int j = 0; j < blockYNums; ++j)
        {
            bottomRightY = std::min(topLeftY + (int)blockHeight - 1, src.rows - 1);
            height = bottomRightY - topLeftY + 1;
            tmpBlockImg = YCbCrImg(cv::Rect(topLeftX, topLeftY, width, height));
            topLeftY += blockHeight;
            Block tmpBlock(tmpBlockImg);
            getMbMrDbDr(tmpBlock);
            // Db, Dr小到一定程度则舍去该block（小的程度为多少论文好像没提及）
            if (tmpBlock.Db - 6.6 < 0.0000001 && tmpBlock.Dr - 6.6 < 0.0000001)
                continue;
            blocks.push_back(tmpBlock);
        }
        topLeftX += blockWidth;
    }
    // 计算全局Mb, Mr, Db, Dr
    double gMb = 0.0, gMr = 0.0, gDb = 0.0, gDr = 0.0;
    for (auto &b : blocks)
    {
        gMb += b.Mb;
        gMr += b.Mr;
        gDb += b.Db;
        gDr += b.Dr;
    }
    gMb /= blocks.size();
    gMr /= blocks.size();
    gDb /= blocks.size();
    gDr /= blocks.size();
    // 根据公式找白点
    vector<int> YHist(256);
    int whitePointsCnt = 0;
    double Ymax = 0.0;
    vector<cv::Point> whitePoints;
    whitePoints.reserve(YCbCrImg.total() / 2);
    // 提前计算出两个公式左右两边的常量
    double tmpLeft1 = gMb + gDb * mySignFunc(gMb) + 128.0;
    double tmpLeft2 = 1.5 * gMr + gDr * mySignFunc(gMr) + 128.0;
    double tmpRight1 = 1.5 * gDb;
    double tmpRight2 = 1.5 * gDr;
    for (auto i = 0; i < YCbCrImg.rows; i++)
    {
        for (auto j = 0; j < YCbCrImg.cols; j++)
        {
            if (Ymax < double(YCbCrImg.at<Vec3b>(i, j)[0]))
                Ymax = double(YCbCrImg.at<Vec3b>(i, j)[0]);
            if (std::abs(double(YCbCrImg.at<Vec3b>(i, j)[1]) - tmpLeft1) < tmpRight1 &&
                std::abs(double(YCbCrImg.at<Vec3b>(i, j)[2]) - tmpLeft2) < tmpRight2)
            {
                YHist[YCbCrImg.at<Vec3b>(i, j)[0]]++;
                // Mat::at(cv::Point)时，x - col，y - row，需要转换
                whitePoints.push_back(cv::Point(j, i));
            }
        }
    }
    // 计算前10%的阈值
    int YT = 0;
    whitePointsCnt = whitePoints.size();
    int tmpWhitePointCnt = 0;
    for (auto i = 255; i >= 0; i--)
    {
        tmpWhitePointCnt += YHist[i];
        YT = i;
        if (tmpWhitePointCnt >= whitePointsCnt / 10)
            break;
    }
    // 计算前10%亮度白点的平均BGR，并计算出增益
    // 注意：如果点数太多的话，可能会导致int64溢出，应换其他方法求和
    int64 sumB = 0, sumG = 0, sumR = 0;
    whitePointsCnt = 0;
    for (auto &p : whitePoints)
    {
        if (YCbCrImg.at<Vec3b>(p)[0] < YT)
            continue;
        whitePointsCnt++;
        sumB += src.at<Vec3b>(p)[0];
        sumG += src.at<Vec3b>(p)[1];
        sumR += src.at<Vec3b>(p)[2];
    }
    double avgB, avgG, avgR;
    avgB = double(sumB) / double(whitePointsCnt);
    avgG = double(sumG) / double(whitePointsCnt);
    avgR = double(sumR) / double(whitePointsCnt);
    double kB = Ymax / avgB, kG = Ymax / avgG, kR = Ymax / avgR;
    // 防止溢出，令大于255的像素点为255
    for (size_t i = 0; i < src.total() * 3; i += 3)
    {
        dst.data[i] = std::min(int(double(dst.data[i])*kB), 255);
        dst.data[i + 1] = std::min(int(double(dst.data[i + 1])*kG), 255);
        dst.data[i + 2] = std::min(int(double(dst.data[i + 2])*kR), 255);
    }
}

int main()
{
    Mat input = imread("F://Test_Img//WB2.jpg");
    Mat GW_output, PR_output, ABW_output;
    grayWorld(input, GW_output);
    perfectReflector(input, 0.1, PR_output);
    automaticWhiteBalance(input, ABW_output, 80, 80);
    imshow("ori img", input);
    imshow("gray world res", GW_output);
    imshow("perfect reflector res", PR_output);
    imshow("dynamic method res", ABW_output);
    waitKey(0);
    return 0;
}