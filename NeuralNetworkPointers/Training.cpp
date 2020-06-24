#include "Training.h"
#include <cstdlib>

//contains all activation functions with their derivarives
Training::Training()
{
	epochs = NULL;
	error = NULL;
	MeanSquaredError = NULL;
}

Training::~Training()
{
}

double Training::fncSigmoid(double n)
{
	return 1.0 / (1.0 + exp(-n));
}

double Training::funcSinc(double n)
{
	double x = NULL;
	if (n != 0)
		x = sin(n) / n;
	else if (n == 1.0)
		x = 0;
	return x;
}

double Training::funcSwish(double n) //alternative to relu
{
	double x = NULL;
	x = n / (1.0 + exp(-n));//for debug
	return n / (1.0 + exp(-n));
}

void Training::funcSoftmax()
{

	output_data1 = exp(softmaxVal_1) / (exp(softmaxVal_1) + exp(softmaxVal_2));
	output_data2 = exp(softmaxVal_2)/ (exp(softmaxVal_2) + exp(softmaxVal_1));
}

double Training::fncSigmoidDerivative(double n)
{
	double x = NULL;
	x = n*(1 - n);
	return x;
}

double Training::funcSincDerivative(double n)
{
	double x = NULL;
	if (n != 0)
		x = cos(n) / n - sin(x) / pow(n, 2);
	else if (n == 0)
		x = 0.0;
	return x;
}

double Training::funcSwishDerivative(double n)
{
	double x = NULL;
	x = n * (1.0 / (1.0 + exp(n)));
	return x;
}
///////////////////////////////////////////////////////////////////////////////////////////////
double Training::funcCrossFunctionDerivative(double network_output, double actual_output)
{
	double res = 2 * (actual_output - network_output);
	return res;
}

void Training::cross_entropy_derivative()
{
	double x = -1.0*((label_data.at(0) * (1.0/ output_data1))+(1.0 - label_data.at(0))*(1.0/(1.0 - output_data1)));
	double y = -1.0* ((label_data.at(1) * (1.0 / output_data2)) + (1.0 - label_data.at(1)) * (1.0 / (1.0 - output_data2)));
	output_error.push_back(x);//output error 1
	output_error.push_back(y);//output error 2
}

void Training::softmax_derivative()
{
	double x1 = (exp(softmaxVal_1) * exp(softmaxVal_2)) / pow((exp(softmaxVal_1) + exp(softmaxVal_2)), 2.0);
	double x2 = (exp(softmaxVal_2) * exp(softmaxVal_2)) / pow((exp(softmaxVal_1) + exp(softmaxVal_2)), 2.0);
	softmax_derivative_sum = x1 + x2;
	softmax_derivative_values[0] = (x1); //derivative of output with respect to input
	softmax_derivative_values[1] = (x2); //derivative of output with respect to input
}

void Training::categorical_crossentropy()
{

	/*double x1 = ((label_data.at(0) * log(output_data1)) + ((1.0 - label_data.at(0) * log((1.0 - output_data1)))));
	double x2 = ((label_data.at(1) * log(output_data2)) + ((1.0 - label_data.at(1) * log((1.0 - output_data2)))));*/
	
	double x1 = -(((label_data.at(0) * log(output_data1)) + ((1.0 - label_data.at(0) * log((1.0 - output_data1))))) + ((label_data.at(1) * log(output_data2)) + ((1.0 - label_data.at(1) * log((1.0 - output_data2))))));
	categorical_crossentropy_value = x1; //error of entire network
}

void Training::MSE()
{

}
