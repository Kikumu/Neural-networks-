#pragma once
#include "Layer.h"
#include <vector>

class outputLayer
{
public:
	outputLayer();
	~outputLayer();

	//each feature will have its own weight
	vector<vector<double>>features;
	void calc();
	void calc1();

	double s1;
	double s2;
private:

};


