#include "Training.h"
#include <cstdlib>

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
