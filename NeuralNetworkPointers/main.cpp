
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "outputLayer.h"
#include "Matrix.h"
#include "Training.h"
#include "Eigen/Core"
#include "LayerMaxPooling.h"
Training t;

using namespace std;
using namespace cv;

LayerMaxPooling maxpooling;
//using namespace Eigen;
int main(int argc, char** argv) {

	//im starting with kaggle cats and dogs for c++
	Eigen::Matrix<double, 100, 100>input;
	Mat img_gray;
	img_gray = imread("1.jpg");
	Mat img;
	cvtColor(img_gray, img, cv::COLOR_BGR2GRAY);
	/*for (int i = 0; i != img.cols; i++) {
		for (int j = 0; j != img.rows; j++) {
		   input(i,j) = +(img_gray.at<char>(i, j));
		}
	}*/
	double darray[100][100];//meaning its taking in 90000 elements
	for (int i = 0; i != img_gray.cols; i++) {
		for (int j = 0; j != img_gray.rows; j++) {
			darray[i][j] = +(img_gray.at<char>(i, j)); //if you encounter an error during transfer of image data its probably here
		}
	}
	//double**resFromFuncPrototype = maxpooling.resultant1(5, darray); //already passed vals
	double** resFromFuncPrototype1 = maxpooling.resultant2(3, darray); //already passed vals
	//print array
	for (int r = 0; r < 2; r++) {
		for (int c = 0; c < 2; c++) {
			cout << resFromFuncPrototype1[r][c]; 
		}
		cout << "\n";
	}
	//maxpooling(5, input);
	/*Eigen::Matrix<double, Dynamic, Dynamic>res;
	res = (maxpooling.resultant(5, input));*/
	//res = Eigen::Map<Eigen::MatrixXd>(maxpooling.resultant(5, input));
	//maxpooling.resultant(5, input);
	waitKey(0);
	return 0;
}