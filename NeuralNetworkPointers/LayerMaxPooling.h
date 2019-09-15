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
   

	Eigen::Matrix<double, Dynamic, Dynamic>resultant (int, Eigen::Matrix<double, Dynamic, Dynamic>);
	//double[][] 
	//Matrix<double, Dynamic, Dynamic>Filter; //should have same number of rows and columns
	//Matrix<double, Dynamic, Dynamic>Resultant;
	double** resultant1(int, double[][100]);
	double** resultant2(int, double[][100]);
	int stride;

	// i want a function that returns a matrix
	//MatrixXd resize(int, MatrixXd);
};
