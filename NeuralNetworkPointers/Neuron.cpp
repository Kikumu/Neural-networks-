#include "Neuron.h"

Neuron::Neuron()
{
	sensibility = NULL;
	error = NULL;
	outputValue = NULL;
	//neuronWeightsIn = NULL;
}

Neuron::~Neuron()
{
}


double Neuron::randomizeWeights()
{
	return ((double)rand() / RAND_MAX);
}

void Neuron::addWeightsOut(weights *x)
{
	neuronWeightsOut.push_back(x);
}

void Neuron::addWeightsIn(weights *x)
{
	neuronWeightsIn.push_back(x);
}


vector<weights*> Neuron::getWeightsIn()
{
	return neuronWeightsIn;
}

vector<weights*> Neuron::getWeightsOut()
{
	return neuronWeightsOut;
}


