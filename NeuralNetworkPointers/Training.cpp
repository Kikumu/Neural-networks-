#include "Training.h"

Training::Training()
{
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
	int x = NULL;
	if (n != 0)
		x = sin(n) / n;
	else if (n == 1)
		x = 0;
	return x;
}

double Training::funcSwish(double n) //alternative to relu
{
	int x = NULL;
	x = n / (1.0 + exp(-n));
	return x;
}

double Training::fncSigmoidDerivative(double n)
{
	int x = NULL;
	x = n*(1 - n);
	return x;
}

double Training::funcSincDerivative(double n)
{
	int x = NULL;
	if (n != 0)
		x = cos(n) / n - sin(x) / pow(n, 2);
	else if (n == 0)
		x = 0;
	return x;
}

double Training::funcSwishDerivative(double n)
{
	int x = NULL;
	x = n * (1.0 / (1.0 + exp(n)));
	return x;
}
