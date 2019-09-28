#pragma once
#include "Eigen/Core"
#include <vector>
//typedef<Eigen::Matrix<double, int, int>>
using namespace std;

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
	void poolConv(vector<double>);
	void poolConv2(vector<double>);
	vector<vector<double>>pooledConv;
	vector<vector<double>>pooledConv1;
	//double** poolConvolve(vector<std::vector<double>>);
	int stride;
};
