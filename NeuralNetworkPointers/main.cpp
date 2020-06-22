
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "outputLayer.h"
#include "Training.h"
#include "Eigen/Core"
#include "LayerMaxPooling.h"
#include "Layer.h"
#include "Convolve.h"
#include "CostFunction.h"
#include "BackPropagation.h"

using namespace std;
using namespace cv;

LayerMaxPooling maxpooling;
Layer LayerFunc;
Convolve conv;
outputLayer output;
Training trn;
CostFunction cst;

const double learning_rate = 0.6;
//using namespace Eigen;
int main(int argc, char** argv) {
	int epochs = 0;
	LayerFunc.learning_rate = learning_rate;


	while (epochs < 5) {
		int k = 0;
		//im starting with kaggle cats and dogs for c++
		Mat img_gray;
		img_gray = imread("1.jpg");
		Mat img;
		cvtColor(img_gray, img, cv::COLOR_BGR2GRAY);
		double darray[100][100];

		//INPUT
		for (int i = 0; i != img_gray.cols; i++) {
			for (int j = 0; j != img_gray.rows; j++) {
				darray[i][j] = +(img_gray.at<char>(i, j)); //if you encounter an error during transfer of image data its probably here
			}
		}

		//CONVOLUTIONS AND MAXPOOLING(to match feature map size)
		conv.convole1(darray); //retouched
		for (int i = 0; i < 4; i++)
			maxpooling.poolConv(conv.featureMapData1[i]); //retouched
		for (int i = 0; i < 4; i++)
			conv.convolve2(maxpooling.pooledConv[i]); //retouched
		for (int i = 0; i < 16; i++)
			maxpooling.poolConv2(conv.featureMapData2[i]);//retouched
		for (int i = 0; i < 16; i++)
			conv.convolve3(maxpooling.pooledConv1[i]);

		//FLATTENING
		conv.flatten();
		LayerFunc.forwardPropagate(conv.Flattened_features);
		LayerFunc.forwardPropagate2(LayerFunc.firstLayerData);
		LayerFunc.forwardPropagate3(LayerFunc.secondLayerData); //inputs

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
		trn.label_data.push_back(label[0]);
		trn.label_data.push_back(label[1]);
		cst.actual_output1.push_back(label[0]);
		cst.actual_output1.push_back(label[1]);

		//PREDICTIONS
		cout << "Predictions: ";
		cout << "\n";
		//cout << (trn.output_data1) * 100;
		cout << (trn.output_data1);
		cout << "% sure its a dog";
		cout << "\n";
		//cout << (trn.output_data2)*100;
		cout << (trn.output_data2);
		cout << "% sure its a cat";

		//COST
		cout << "\n";
		cout << "\n";
		cst.costRes_1();
		trn.categorical_crossentropy();
		trn.MeanSquaredError = cst.costdat;
		cout << "Cost: ";
		cout << "\n";
		cout << cst.costdat;
		waitKey(0);

		//BACK PROPAGATION
		//THOUGHT SKELETON: GRAB ALL WEIGHT DATA, COMPUTE DERIVATIVES
		cst.cost_derivative();
		trn.softmax_derivative();
		cout << "\n";
		cout << cst.cost_derivative_data;
		LayerFunc.backpropagation(cst.cost_derivative_data, trn.softmax_derivative_sum);

		
		epochs++;
	}
	return 0;
}