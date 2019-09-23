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
	for (int i = 0; i != 100; i++) {
		for (int j = 0; j != 100; j++) {
			inputTensor(i, j) = input[i][j]; //matrix for takin in input
		}
	}
	vector<double>maxNumberStore;
	Eigen::Matrix<double, 40, 40>filter;
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

void LayerMaxPooling::poolLayerby40(double**)
{
}




