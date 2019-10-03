#include "CostFunction.h"
#include <cstddef>
#include "Eigen/Core"
#include "Layer.h"
Layer layer_func;


CostFunction::CostFunction()
{
}

CostFunction::~CostFunction()
{
}

void CostFunction::costRes()
{
	Eigen::Vector2d actual_output;
	Eigen::Vector2d network_output;
	Eigen::Vector2d difference;
	double length;
	//double costData1;
	for (int i = 0; i < 2; i++) {
				network_output(i) = network_output1[i]; //copied layer output to eigen
				actual_output(i) = actual_output1[i];
		}
	difference = actual_output - network_output;
	length = difference.squaredNorm();
	double sum = 0;
	double inputs = 2.0;
	for (int j = 0; j < 2; j++) {
		sum += length;
	}
	costdat = (1.0 /(2.0*inputs))*sum;

	//layer_func.costData.push_back(costData1);
}

//double CostFunction::costRes(int a , int b, double actual[2], double output[2])
//{
//	Eigen::Vector2d actual_output;  //gather from labelling but for now
//	Eigen::Vector2d network_output;
//	for (int i = 0; i < 2; i++) {
//		network_output(i) = output[i]; //copied layer output to eigen
//	}
//
//	//find out how to pin actual output to input
//
//	double cost = NULL;
//	//cost = 1/a()
//	return 0.0;
//}

