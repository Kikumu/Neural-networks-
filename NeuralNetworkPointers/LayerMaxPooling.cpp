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


void LayerMaxPooling::poolConv(vector<double> e, int epochs)
{
	Eigen::Matrix<double, 19, 19>inputTensor;
	vector<double>maxNumberStore;
	Eigen::Matrix<double, 5, 5>filter;

	//INPUT VECTOR MUST BE CONVERTED BACK TO EIGEN MATRIX TO BE WORKED ON SO OUR INPUT IS NOW INPUT TENSOR
	int k = 0;
	for (int i = 0; i != 19; i++) {
		for (int j = 0; j != 19; j++) {
			inputTensor(i, j) = e[k]; 
			++k;
		}
	}
	
	//MAX VALUE IS FOUND PER 5 BY 5 BLOCK FILTER
	for (int r = 0; r < 19; r++) {
		//LIMITED TO 15 FOR PADDING PURPOSES TO AVOID MATRIX SIZE MISMATCH
		if (r < 15) {
			for (int c = 0; c < 19; c++) {
				if (c < 15) {
					filter = inputTensor.block(r, c, 5, 5);
					maxNumberStore.push_back(filter.mean()); //changed to average pooling
				}
			}
		}
	}

	if (epochs == 0) {
		//RETURNED IN THE FORM OF A VECTOR
		pooledConv.push_back(maxNumberStore);
	}
	else {
		pooledConv[0] = maxNumberStore;
	}
	
}

void LayerMaxPooling::poolConv2(vector<double>e, int epochs)
{
	Eigen::Matrix<double, 13, 13>inputTensor;
	vector<double>maxNumberStore;
	Eigen::Matrix<double, 5, 5>filter;

	//VECTOR HAS TO BE RECONVERTED BACK TO A MATRIX
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
					maxNumberStore.push_back(filter.mean()); //changed to average pooling
				}
			}
		}
	}


	if (epochs == 0) {
		//RETURNED IN THE FORM OF A VECTOR
		pooledConv1.push_back(maxNumberStore);
	}
	else {
		pooledConv1[0] = maxNumberStore;
	}
	//pooledConv1.push_back(maxNumberStore);
}







