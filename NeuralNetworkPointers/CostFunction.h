#pragma once
#include <vector>

using namespace std;
class CostFunction
{
public:
	CostFunction();
	~CostFunction();

	vector<double>network_output1;
	vector<double>actual_output1;
	double actual_out[2];      //actual output(label) data
	double network_out[2];     //network output data
	double derivative_cost[2]; //derivative of each output
	//variables: total number of inputs(per var),input as a whole, current network output, actual output(i need to find some sort of label)

	//output data
	double o1;
	double o2;

	//derivative output data
	double d_o1;
	double d_o2;


	void costRes();
	void costRes_1();
	

	//derivative
	void cost_derivative_per_out();


	//overall derivative
	void cost_derivative();

	//overall cost
	double costdat;
	double cost_derivative_data;


	//getters
	double return_cost_derivative();
private:

};

