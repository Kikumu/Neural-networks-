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
	/*double sum = 0;
	double inputs = 2.0;
	for (int j = 0; j < 2; j++) {
		sum += length;
	}*/
	//costdat = (1.0 /(2.0*inputs))*sum;
	costdat = length;

	//layer_func.costData.push_back(costData1);
}

