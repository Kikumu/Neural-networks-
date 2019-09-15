#include "Layer.h"

Layer::Layer(int i, int h):number_of_inputs_size(i),number_of_output_size(h) {
	number_of_inputs_size = NULL;
	number_of_output_size = NULL;
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

void Layer::forwardPropagate(Matrix)
{
}

void Layer::init(double mu, double sigma)
{
}

void Layer::backpropagation()
{
}


