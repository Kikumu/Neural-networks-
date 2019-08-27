#pragma once
#include "weights.h"
#include <vector>

using namespace std;

class weights;
//all public because these values will keep changing/be changed by different layer to layer association
class Neuron
{
public:
	Neuron();
	~Neuron();
	double randomizeWeights();
	//getters
	double outputValue;
	double error;
	double sensibility;
	void addWeightsOut(weights *);
	void addWeightsIn(weights *);
	vector<weights*>neuronWeightsIn;
	vector<weights*>neuronWeightsOut;
	vector<weights*>getWeightsIn();
	vector<weights*>getWeightsOut();
private:
	
};

