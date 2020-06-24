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
	void poolConv(vector<double>, int);
	void poolConv2(vector<double>, int);
	vector<vector<double>>pooledConv;
	vector<vector<double>>pooledConv1;
	//double** poolConvolve(vector<std::vector<double>>);
	int stride;
};
