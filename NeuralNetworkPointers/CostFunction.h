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
	//variables: total number of inputs(per var),input as a whole, current network output, actual output(i need to find some sort of label)

	void costRes();
	void costRes_1();
	void cost_derivative();

	double costdat;
	double cost_derivative_data;


	//getters
	double return_cost_derivative();
private:

};

