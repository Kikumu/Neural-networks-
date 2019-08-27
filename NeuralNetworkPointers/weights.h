#pragma once
#include "Neuron.h"
#include <vector>

using namespace std;

class Neuron;
// a weight can only have 2 neurons connected on each end
class weights
{
public:
	weights();
	~weights();
	//void setNeurons(vector<Neuron*>n);
	void SetOneNeuron(Neuron*); // a weight can only have 2 neurons connected on each end
	void setTwoNeuron(Neuron*); // a weight can only have 2 neurons connected on each end
	void setWeight(double);
	double getWeight();
	double weight;
private:
	//vector<Neuron*>Neurons;
	Neuron* one;
	Neuron* Two;
	//double weight;
};
