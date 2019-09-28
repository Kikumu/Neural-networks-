
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
#include "Convolve.h"

using namespace std;
using namespace cv;

LayerMaxPooling maxpooling;
Layer LayerFunc;
Convolve conv;
vector<vector<double>>FeatureConv;

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
	FeatureConv = conv.convole1(darray);
	double** poolLayer = maxpooling.resultant(darray); //already passed vals *60  by 60 from 100 by 100)
    //LayerFunc.forwardPropagate(poolLayer);
	double** poolLayer2 = maxpooling.poolLayerby40(poolLayer);
	//LayerFunc.forwardPropagate2(poolLayer2);
	//dummy label
	double predictions[2];
	predictions[0] = LayerFunc.secondLayerData[0];
	predictions[1] = LayerFunc.secondLayerData[1];
	double label[2];
	label[0] = 0.0;
	label[1] = 1.0;
	LayerFunc.costRes(100.0, 10000.0, predictions, label);
	//create a condition such that if data is from cat clear label and put top 1 and bottom zero and vice versa for dog
	//clear pointers
	cout << "predictions:";
	cout << "\n";
	for (int i = 0; i < 2; i++) {
		cout << predictions[i];
		cout << "\n";
	}
	cout << "\n";
	cout << "\n";
	cout << "Cost data:";
	cout << "\n";
    cout<< LayerFunc.costData.at(0);
	waitKey(0);
	return 0;
}