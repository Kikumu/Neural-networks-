#include "Neuron.h"

Neuron::Neuron()
{
	sensibility = NULL;
	error = NULL;
	outputValue = NULL;
}

Neuron::~Neuron()
{
}

void Neuron::setOutputValue(double x)
{
	outputValue =  x;
}

void Neuron::setError(double x)
{
	error = x;
}

void Neuron::setSensibility(double x)
{
	sensibility = x;
}

void Neuron::setWeightsIn(vector<weights*> x)
{
	neuronWeightsIn = x;
}

void Neuron::setWeightsOut(vector<weights*> x)
{
	neuronWeightsOut = x;
}

void Neuron::addWeightIn(weights*x)
{
	neuronWeightsIn.push_back(x);
}

double Neuron::randomizeWeights()
{
	return ((double)rand() / RAND_MAX);
}

vector<weights*> Neuron::getWeightsIn()
{
	return neuronWeightsIn;
}

vector<weights*> Neuron::getWeightsOut()
{
	return neuronWeightsOut;
}


