#pragma once

#include <vector>
#include "Neuron.h"

using namespace std;

class Neuron;
class weights;

class Layer
{
public:
	Layer();
	~Layer();

	void setNumberOfNeurons(vector<Neuron*>);
	int getNumberOfNeurons(int);
	vector<Neuron*>getNeurons();
private:
	vector<Neuron*>LayerNeurons;
};


