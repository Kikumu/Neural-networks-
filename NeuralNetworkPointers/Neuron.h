#pragma once
#include "weights.h"
#include <vector>

using namespace std;

class weights;

class Neuron
{
public:
	Neuron();
	~Neuron();
	void setOutputValue(double);
	void setError(double);
	void setSensibility(double);
	void setWeightsIn(vector<weights*>);
	void setWeightsOut(vector<weights*>);
	void addWeightIn(weights*);
	double randomizeWeights();
	//getters

	vector<weights*>getWeightsIn();
	vector<weights*>getWeightsOut();
private:
	double outputValue;
	double error;
	double sensibility;
	vector<weights*>neuronWeightsIn;
	vector<weights*>neuronWeightsOut;
};

