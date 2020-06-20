#include "LayerAvgPooling.h"
#include "Eigen/Core"
#include <vector>


using namespace std;

LayerAvgPooling::LayerAvgPooling()
{
}

LayerAvgPooling::~LayerAvgPooling()
{
}

double** LayerAvgPooling::resultant(double input[][100])
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
					maxNumberStore.push_back(filter.mean());
				}
			}
		}
	}

	int k = 0;
	convdat = new double* [60];
	for (int r = 0; r < 60; r++) {
		convdat[r] = new double[60];
		for (int c = 0; c < 60; c++) {
			convdat[r][c] = maxNumberStore[k];
			++k;
		}
	}
	return convdat;
}
