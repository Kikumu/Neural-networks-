#pragma once
#include <vector>
using namespace std;

class Training
{
public:
	Training();
	~Training();

	int epochs;
	double error;
	double MeanSquaredError;
	double categorical_crossentropy_value;
	//Activation functions for forward propagation
	double fncSigmoid(double);
	double funcSinc(double);
	double funcSwish(double);
	void funcSoftmax();

	//derivative for back propagation
	double fncSigmoidDerivative(double);
	double funcSincDerivative(double);
	double funcSwishDerivative(double);
	void cross_entropy_derivative();

	//softmax vals
	double softmaxVal_1;
	double softmaxVal_2;

	//probabilities
	double output_data1;
	double output_data2;
	
	//values for categorical crossentropy
	vector<double>label_data;


	//loss
	void categorical_crossentropy();
private:

};
