#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	//Doc va hien thi anh
	Mat img = imread("E:\\D13CNPM4\\ImageProcessing\\Image\\lena.png", CV_LOAD_IMAGE_COLOR);
	imshow("orginal", img);


	//hien thi thong tin anh
	cout << img.rows << endl;
	cout << img.cols << endl;


	//truy cap cao phan tu anh
	cout << (int)img.at<Vec3b>(100, 150)[2] << endl;
	imshow("red", img);


	//tao anh rong
	 Mat dst(100, 150, CV_8UC3,Scalar(0,255,0));//Mat dst(100,150,CV_8UC3,Scalar(255,255,255));
	imshow("green", dst);


	//thay doi co va crop anh
	resize(img, dst, Size(200, 100));
	imshow("Resize anh", dst);


	//??Ve hcn len anh nguon
	Rect r(100, 100, 200, 150);
	dst = img(r);
	rectangle(img, r, Scalar(0, 0, 0), 1);
	imshow("VeHCN", img);


	//Mo hinh mau bang ham
	cvtColor(img, dst, COLOR_RGB2GRAY);
	imshow("color", dst);


	//mo hinh mau bang cong thuc Val=0.3*R+0.5*G+0.2*B
	 dst = img.clone();
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			dst.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0] * 0.2 + img.at<Vec3b>(i, j)[1] * 0.5 + img.at<Vec3b>(i, j)[2] * 0.3;
			dst.at<Vec3b>(i, j)[0] = dst.at<Vec3b>(i, j)[1];
			dst.at<Vec3b>(i, j)[0] = dst.at<Vec3b>(i, j)[2];
		}
	}
	imshow("cvtColor", dst);

	//tach cac kenh mau RGB
	Mat kenh[3];
	split(img, kenh);
	imshow("blue", kenh[0]);




	//tach cac kenh mau bang ham
	cvtColor(img, dst, COLOR_RGB2HSV);
	split(dst, kenh);
	imshow("red", kenh[2]);


	// hop cac kenh
	split(img, kenh);
	kenh[1] = kenh[2] = 0;
	merge(kenh, 3, dst);
	imshow("HOP", dst);

	//bien doi am ban
	split(img, kenh);
	dst = 255 - kenh[2];
	imshow("amban", dst);


	//tang giam do sang
	split(img, kenh);
	int c = 100;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0] + c;
			img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j)[1] + c;
			img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2] + c;
		}
	}
	imshow("dosang", img);

	//tang giam do tuong phan
	
	for (int i = 0; i < img.rows; i++)
	{
		for(int j=0;j<img.cols;j++)
		{ 
			img.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(img.at<Vec3b>(i, j)[0] - c);
			img.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(img.at<Vec3b>(i, j)[1] - c);
			img.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(img.at<Vec3b>(i, j)[2] - c);
		}
	}
	imshow("tuongphan", img);


	//historam can bang
	imshow("histogram old", kenh[0]);
	equalizeHist(kenh[0], kenh[0]);
	imshow("histogram new", kenh[0]);



	//phan nguong
	imshow("phan nguong old", kenh[0]);
	threshold(kenh[0], kenh[0], 100, 255, THRESH_BINARY);
	imshow("phan nguong new", kenh[0]);


	//hien thi diem lan can
	int i = 100, j = 150;
	for (int s = 0; s < 2; s++)
	{
		for (int t = 0; t < 2; t++)
		{
			cout << (int)img.at<Vec3b>(i + s, j + t)[0] << ",";
			cout << (int)img.at<Vec3b>(i + s, j + t)[1] << ",";
			cout << (int)img.at<Vec3b>(i + s, j + t)[2] << ",";

		}
	}
	cout << endl;


	//nhan chap filter
	//++loc trung binh
	blur(img, dst, Size(3, 3), Point(-1, -1), 4);
	imshow("loc trung binh", dst);
	medianBlur(img, dst, 3);
	imshow("loc trung vi", dst);


	//nhan chap filter(khai bao ham khi chua co san)
	float h[9] = { 1 / 10,1 / 10,1 / 10,
		       	1 / 10,2 / 10,1 / 10,
				1 / 10,1 / 10,1 / 10 };
	Mat kernel(3, 3, CV_32S, h);
	filter2D(img, dst, -1, kernel, Point(-1, -1), 0, 4);
	imshow("filter2D", dst);



	//Tim bien anh sobel theo huong x
	Mat dstx, dsty;
	Sobel(img, dstx, CV_64F, 1,0, 3);
	imshow("sobel x", dstx);



	//tim bien anh sobel theo huong y
	Sobel(img, dsty, CV_64F, 0, 1, 3);
	imshow("sobel y", dsty);


	//timbien theo 2 huong
	dst = abs(dstx) + abs(dsty);
	imshow("2 huong x,y", dst);

	//tim bien anh theo ham Canny
	double t1 = 30, t2 = 200;
	Canny(img, dst, t1, t2,3, false);
	imshow("canny", dst);


	//tim bien anh theo ham Lapplace
	Laplacian(img, dst, -1, 1, 1, 0, 4);
	imshow("lapplace", dst);



	//chuyen anh mau sang anh xam theo cong thuc : R*x+G*y+B*z
	int x=50, y=100, z=150;
	Mat gray(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
	imshow("anh gray old", gray);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			gray.at<uchar>(i, j) = img.at<Vec3b>(i, j)[2] * x + img.at<Vec3b>(i, j)[1] * y + img.at<Vec3b>(i, j)[0] * z;
		}
	}
	imshow("anh gray", gray);






		//tim muc xam min max tren 1 kenh (gray0
			double min, max;
			minMaxLoc(gray, &min, &max);
		cout << "max=" << max << endl;
	waitKey();
	return 0;




}