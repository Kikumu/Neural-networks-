
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
#include "outputLayer.h"
#include "CostFunction.h"

using namespace std;
using namespace cv;

LayerMaxPooling maxpooling;
Layer LayerFunc;
Convolve conv;
outputLayer output;
Training trn;
CostFunction cst;
//using namespace Eigen;
int main(int argc, char** argv) {

	int k = 0;
	//im starting with kaggle cats and dogs for c++
	Mat img_gray;
	img_gray = imread("1.jpg");
	Mat img;
	cvtColor(img_gray, img, cv::COLOR_BGR2GRAY);
	double darray[100][100];
	for (int i = 0; i != img_gray.cols; i++) {
		for (int j = 0; j != img_gray.rows; j++) {
			darray[i][j] = +(img_gray.at<char>(i, j)); //if you encounter an error during transfer of image data its probably here
		}
	}
	conv.convole1(darray);
	for(int i = 0; i < 3; i++)
		maxpooling.poolConv(conv.featureMapData1[i]);
	for(int i = 0; i < 3; i++)
	    conv.convolve2(maxpooling.pooledConv[i]);
	for(int i = 0; i < 12; i++)
	    maxpooling.poolConv2(conv.featureMapData2[i]);
	for (int i = 0; i < 12; i++)
		conv.convolve3(maxpooling.pooledConv1[i]);

	conv.flatten();
	LayerFunc.forwardPropagate(conv.Flattened_features);
	LayerFunc.forwardPropagate2(LayerFunc.firstLayerData);
	LayerFunc.forwardPropagate3(LayerFunc.secondLayerData);
	//probabilities
	trn.softmaxVal_1 = LayerFunc.ThirdWeightData.at(0);
	trn.softmaxVal_2 = LayerFunc.ThirdWeightData.at(1);
	trn.funcSoftmax();
	cst.network_output1.push_back(trn.output_data1);
	cst.network_output1.push_back(trn.output_data2);
   //LABEL
	double label[2];
	label[0] = 0.0;
	label[1] = 1.0;
	cst.actual_output1.push_back(label[0]);
	cst.actual_output1.push_back(label[1]);
	//PREDICTIONS
	cout << "Predictions: ";
	cout << "\n";
	cout << trn.output_data1;
	cout << "\n";
	cout << trn.output_data2;

	//COST
	cout << "\n";
	cout << "\n";
	cst.costRes();
	cout << "Cost: ";
	cout << "\n";
	cout << cst.costdat;

	waitKey(0);
	return 0;
}