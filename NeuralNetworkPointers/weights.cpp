#include "weights.h"

weights::weights()
{
	one = NULL;
	Two = NULL;
	weight = NULL;
}

weights::~weights()
{
}

void weights::SetOneNeuron(Neuron * x)
{
	one = x;
}

void weights::setTwoNeuron(Neuron * x)
{
	Two = x;
}

void weights::setWeight(double x)
{
	weight = x;
}

double weights::getWeight()
{
	return weight;
}

