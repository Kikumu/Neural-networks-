
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
#include "Layer.h"
Training t;

using namespace std;
using namespace cv;

LayerMaxPooling maxpooling;
Layer LayerFunc;
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
	double** poolLayer = maxpooling.resultant(darray); //already passed vals *60  by 60 from 100 by 100)
    LayerFunc.forwardPropagate(poolLayer);
	double** poolLayer2 = maxpooling.poolLayerby40(poolLayer);
	LayerFunc.forwardPropagate2(poolLayer2);
	vector<double>predictions = LayerFunc.secondLayerData;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			cout << poolLayer2[i][j];
			cout << " ";
		}
		cout << "\n";
	}
	//pool more 40 by 40
	//double**poolLayer2 =


	







	waitKey(0);
	return 0;
}