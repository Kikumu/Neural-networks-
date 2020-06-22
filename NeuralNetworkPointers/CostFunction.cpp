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
	costdat = length;
}

void CostFunction::costRes_1()
{
	double res = (1.0/(2.0 * 2.0));
	double out1;
	out1 = actual_output1[0] - network_output1[0];
	double out2;
	out2 = actual_output1[1] - network_output1[1];
	double res_1 = pow(out1, 2.0);
	double res_2 = pow(out2, 2.0);
	double res_3 = res * (res_1 + res_2);
	costdat = res_3;
}

 void CostFunction::cost_derivative()
{
	double out1;
	out1 = abs(actual_output1[0] - network_output1[0]);
	double out2;
	out2 = abs(actual_output1[1] - network_output1[1]);
	double res_1 = 2.0 * (out1);
	double res_2 = 2.0 * (out2);
	double res_3 = (res_1 + res_2);
	cost_derivative_data = res_3;
}

double CostFunction::return_cost_derivative()
{
	return cost_derivative_data;
}

