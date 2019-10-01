#pragma once

using namespace std;

class Training
{
public:
	Training();
	~Training();

	int epochs;
	double error;
	double MeanSquaredError;

	//Activation functions for forward propagation
	double fncSigmoid(double);
	double funcSinc(double);
	double funcSwish(double);
	double funcSoftmax(double);
	//derivative for back propagation
	double fncSigmoidDerivative(double);
	double funcSincDerivative(double);
	double funcSwishDerivative(double);
	
private:

};
