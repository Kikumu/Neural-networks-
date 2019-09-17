
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
	Mat img_gray;
	img_gray = imread("1.jpg");
	Mat img;
	cvtColor(img_gray, img, cv::COLOR_BGR2GRAY);
	double darray[100][100];//meaning its taking in 90000 elements
	for (int i = 0; i != img_gray.cols; i++) {
		for (int j = 0; j != img_gray.rows; j++) {
			darray[i][j] = +(img_gray.at<char>(i, j)); //if you encounter an error during transfer of image data its probably here
		}
	}
	double** resFromFuncPrototype3 = maxpooling.resultant(darray); //already passed vals
	//print array
	for (int r = 0; r < 60; r++) {
		for (int c = 0; c < 60; c++) {
			cout << resFromFuncPrototype3[r][c]; 
			cout << " ";
		}
		cout << "\n";
	}
	waitKey(0);
	return 0;
}