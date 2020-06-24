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

//cost per output neuron
void CostFunction::costRes() //reworked
{
	double res = (1.0 / (2.0 * 2.0)); //const


	double out1;
	out1 = actual_out[0] - network_out[0];
	double res_1 = pow(out1, 2.0);
	double res_3 = res * (res_1);
	o1 = res_3;


	double out2;
	out2 = actual_out[1] - network_out[1];
	double res_2 = pow(out2, 2.0);
	res_3 = res * res_2;
	o2 = res_3;
}

//(Overall network cost)
void CostFunction::costRes_1()
{
	double res = (1.0/(2.0 * 2.0));
	double out1;
	out1 = actual_out[0] - network_out[0];
	double out2;
	out2 = actual_out[1] - network_out[1];
	double res_1 = pow(out1, 2.0);
	double res_2 = pow(out2, 2.0);
	double res_3 = res * (res_1 + res_2);
	costdat = res_3;
}

void CostFunction::cost_derivative_per_out()
{
	double out1;
	out1 = actual_out[0] - network_out[0];
	double out2;
	out2 = actual_out[1] - network_out[1];
	double res_1 = 2.0 * (out1);
	double res_2 = 2.0 * (out2);

	derivative_cost[0] = res_1;
	derivative_cost[1] = res_2;

	d_o1 = res_1;
	d_o2 = res_2;
}





 void CostFunction::cost_derivative()
{
	double out1;
	out1 = abs(actual_out[0] - network_out[0]);
	double out2;
	out2 = abs(actual_out[1] - network_out[1]);
	double res_1 = 2.0 * (out1);
	double res_2 = 2.0 * (out2);
	double res_3 = (res_1 + res_2);
	cost_derivative_data = res_3;
}

double CostFunction::return_cost_derivative()
{
	return cost_derivative_data;
}

