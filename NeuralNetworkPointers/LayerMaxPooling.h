#pragma once
#include "Eigen/Core"

//typedef<Eigen::Matrix<double, int, int>>
using namespace Eigen;

class LayerMaxPooling {
	//stride
	//filtersize
	//padding?
	//typedef Eigen::Matrix<double, Dynamic, Dynamic>Matrix; 

public:
	LayerMaxPooling();
	~LayerMaxPooling();
	double** resultant(double[][100]);
	double** poolLayerby40(double**);
	int stride;
};
