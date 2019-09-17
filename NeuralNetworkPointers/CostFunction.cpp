#include "CostFunction.h"
#include <cstddef>
#include "Eigen/Core"

CostFunction::CostFunction()
{
}

CostFunction::~CostFunction()
{
}

double CostFunction::costRes(int a , int b, double actual[2], double output[2])
{
	Eigen::Vector2d actual_output;  //gather from labelling but for now
	Eigen::Vector2d network_output;
	for (int i = 0; i < 2; i++) {
		network_output(i) = output[i]; //copied layer output to eigen
	}

	//find out how to pin actual output to input

	double cost = NULL;
	//cost = 1/a()
	return 0.0;
}

