
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "inputLayer.h"
#include "weights.h"
#include "hiddenLayer.h"
#include "Neuron.h"
#include "outputLayer.h"
#include "BuildNetwork.h"
#include "Matrix.h"
#include "Training.h"



int  NumberOfInputNeurons = NULL;
int  NumberOfHiddenLayerNeurons = NULL;
int  numberOfOutputNeurons = NULL;
vector<Neuron*>outputLayerNeurons;
vector<Neuron*>hiddenLayerNeurons;
vector<weights*>weight;

Training t;
Neuron n;
//NB: lets first work with a single layered network shall we
const int r = 300;
const int c = 300;
const int c1 = 15;

using namespace std;
using namespace cv;
int main(int, char**) {

	//im starting with kaggle cats and dogs for c++

	//t.fncSigmoid();
	//synthensizing my input layer
	Mat img_gray;
	img_gray = imread("grayImage.png");
	double darray[r][c];//meaning its taking in 90000 elements
	for (int i = 0; i != img_gray.cols; i++) {
		for (int j = 0; j != img_gray.rows; j++) {
			darray[i][j] = +(img_gray.at<char>(i, j)); //if you encounter an error during transfer of image data its probably here
		}
	}
	//TODO: WEIGHTS
	//remember, number of rows is equal to the number of cols
	//weights depend on hidden layer
	//need to find a way to "feed forward" per layer
	//create a function
	// thingds to consider:
	//number of rows of weights must be 
	//there fore a 300(to match number of inputs)rows by (number of hidden neurons you want)columns
	//iinitialize weights
	//i want to feed it to 15 neurons
	double iWeights[r][c1];
	double R = r*r;
	for (int i = 0; i != 300; i++) {
		for (int j = 0; j != 15; j++) {
			iWeights[i][j] = (n.randomizeWeights())/sqrt(R);
			//iWeights[i][j] = (n.randomizeWeights());
			//iWeights[i][j] = (n.randomizeWeights())/sqrt(r);
			//iWeights[i][j] = (n.randomizeWeights()) / pow(R, 0.5);
			//iWeights[i][j] = (n.randomizeWeights()) / pow(r,0.5);
		}
	}
	//remember to multiply each weight and input with an activation function during training
	//result is number of rows as first matrix, same number of columns as second matrix
	int sharedDim = r;
	double h1result[r][c1]; //hidden layer to be calculated;300,15
	
	for (int i = 0; i != 300; i++) {
		for (int j = 0; j != 15; j++) {
			h1result[i][j] = NULL;
		}
	}
	//matmul;
	for (int i = 0; i != 300; i++) {
		for (int j = 0; j != 15; j++) {
			//h1result[i][j] = NULL;
			for (int k = 0; k != sharedDim; k++) {
				double init = darray[i][k] * iWeights[k][j];
				h1result[i][j] += t.fncSigmoid(init);
			}
		}
	}
	//out matrix. 10 entries(numbers 0 to 1)
	//i want only one column
	//or only one row with multiple column
	double h2weights[15][1];
	double h2result[300][1];
	//cost function



	double outputArray1[1][10];
	//double outputWeights[15][10];
	//double outputLayerW[]
	//output after 15?
	//dor product of the matrix

	waitKey(0);
	return 0;
}