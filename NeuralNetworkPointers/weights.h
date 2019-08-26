#pragma once
#include "Neuron.h"
#include <vector>

using namespace std;

class Neuron;

class weights
{
public:
	weights();
	~weights();
	//void setNeurons(vector<Neuron*>n);
	void SetOneNeuron(Neuron*);
	void setTwoNeuron(Neuron*);
	void setWeight(double);
	double getWeight();
private:
	//vector<Neuron*>Neurons;
	Neuron* one;
	Neuron* Two;
	double weight;
};
