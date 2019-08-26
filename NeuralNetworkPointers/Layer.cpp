#include "Layer.h"

Layer::Layer()
{
}

Layer::~Layer()
{
}

void Layer::setNumberOfNeurons(vector<Neuron*> x)
{
	LayerNeurons = x;
}

int Layer::getNumberOfNeurons(int x)
{
	x = LayerNeurons.size();
	return x;
}

vector<Neuron*> Layer::getNeurons()
{
	return LayerNeurons;
}


