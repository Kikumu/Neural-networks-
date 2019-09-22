#include "Layer.h"

Layer::Layer(int i, int h):number_of_inputs_size(i),number_of_output_size(h) {
	number_of_inputs_size = NULL;
	number_of_output_size = NULL;
}

Layer::Layer()
{
}

Layer::~Layer()
{
}

int Layer::getInputSize()
{
	return number_of_inputs_size;
}

int Layer::getOutputSize()
{
	return number_of_output_size;
}

//double** Layer::forwardPropagate(double[][100])
//{
//	//initialise weights
//	//place an rng to each var
//
//	return nullptr;
//}

double** Layer::forwardPropagate(double**)
{
	return nullptr;
}

double Layer::LayerSensitivity()
{
	return 0.0;
}



void Layer::init(double mu, double sigma)
{
}

void Layer::backpropagation()
{
}


