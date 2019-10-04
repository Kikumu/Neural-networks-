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
	void funcSoftmax();

	//derivative for back propagation
	double fncSigmoidDerivative(double);
	double funcSincDerivative(double);
	double funcSwishDerivative(double);

	//softmax vals
	double softmaxVal_1;
	double softmaxVal_2;

	//probabilities
	double output_data1;
	double output_data2;
	
private:

};
