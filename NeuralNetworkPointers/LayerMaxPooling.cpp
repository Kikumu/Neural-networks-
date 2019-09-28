#include "LayerMaxPooling.h"
#include "Eigen/Core"
#include <vector>


using namespace std;

LayerMaxPooling::LayerMaxPooling()
{
	
}

LayerMaxPooling::~LayerMaxPooling()
{
}

double** LayerMaxPooling::resultant(double input[][100])
{
	double** convdat = NULL;
	Eigen::Matrix<double, 100, 100>inputTensor;
	vector<double>maxNumberStore;
	Eigen::Matrix<double, 40, 40>filter;

	for (int i = 0; i != 100; i++) {
		for (int j = 0; j != 100; j++) {
			inputTensor(i, j) = input[i][j]; //matrix for takin in input
		}
	}
	
	for (int r = 0; r < 100; r++) {
		if (r < 60) {
			for (int c = 0; c < 100; c++) {
				if (c < 60) {
					filter = inputTensor.block(r, c, 40, 40);
					maxNumberStore.push_back(filter.maxCoeff());
				}
			}
		}
	}
	int k = 0; //note arrays are a myth pointers are GOATS
	convdat = new double*[60];
	for (int r = 0; r < 60; r++) {
		convdat[r] = new double[60];
		for (int c = 0; c < 60; c++) {
			convdat[r][c] = maxNumberStore[k];
			++k;
		}
	}
	return convdat; //activation map dim * 60
	///are we supposed to multiply conv layer by respective weights hence "activation map" hence neurons formed
}



double** LayerMaxPooling::poolLayerby40(double**i)
{
	//outputsize calc = (inputwidth - filterwidth)/stride + 1
	Eigen::Matrix<double, 60, 60>inputData;
	Eigen::Matrix<double, 20, 20>filter1;
	Eigen::Matrix<double, 2, 2>outData; //activationmap
	vector<double>maxNumberStore;
	double** outData1 = NULL;

	int stride = 5;
	for (int c = 0; c < 60; c++)
	{
		for (int r = 0; r < 60; r++) {
			inputData(r, c) = i[r][c];
		}
	}


	for (int r = 0; r < 60; r++) {
		if (r < 35) {   
			r + stride;
			for (int c = 0; c < 40; c++) {
				if (c < 35) {
					c + stride;
					filter1 = inputData.block(r, c, 20, 20);
					maxNumberStore.push_back(filter1.maxCoeff());
				}
			}
		}
	}

	int k = 0;
	outData1 = new double* [2];
	for (int r = 0; r < 2; r++) {
		outData1[r] = new double[2];
		for (int c = 0; c < 2; c++) {
			outData1[r][c] = maxNumberStore[k];
			++k;
		}
	}

	return outData1;
}

//19 by 19


void LayerMaxPooling::poolConv(vector<double> e)
{
	Eigen::Matrix<double, 19, 19>inputTensor;
	vector<double>maxNumberStore;
	Eigen::Matrix<double, 5, 5>filter;

	int k = 0;
	for (int i = 0; i != 19; i++) {
		for (int j = 0; j != 19; j++) {
			inputTensor(i, j) = e[k]; //matrix for takin in input
			++k;
		}
	}

	for (int r = 0; r < 19; r++) {
		if (r < 15) {
			for (int c = 0; c < 19; c++) {
				if (c < 15) {
					filter = inputTensor.block(r, c, 5, 5);
					maxNumberStore.push_back(filter.maxCoeff());
				}
			}
		}
	}
	pooledConv.push_back(maxNumberStore);
}

void LayerMaxPooling::poolConv2(vector<double>e)
{
	Eigen::Matrix<double, 13, 13>inputTensor;
	vector<double>maxNumberStore;
	Eigen::Matrix<double, 5, 5>filter;

	int k = 0;
	for (int i = 0; i != 13; i++) {
		for (int j = 0; j != 13; j++) {
			inputTensor(i, j) = e[k]; //matrix for takin in input
			++k;
		}
	}

	for (int r = 0; r < 13; r++) {
		if (r < 8) {
			for (int c = 0; c < 13; c++) {
				if (c < 8) {
					filter = inputTensor.block(r, c, 5, 5);
					maxNumberStore.push_back(filter.maxCoeff());
				}
			}
		}
	}
	pooledConv1.push_back(maxNumberStore);
}







