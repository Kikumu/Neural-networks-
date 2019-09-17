#pragma once

class CostFunction
{
public:
	CostFunction();
	~CostFunction();

	//variables: total number of inputs(per var),input as a whole, current network output, actual output(i need to find some sort of label)

	double costRes(int, int, double [2], double [2]);
private:

};

