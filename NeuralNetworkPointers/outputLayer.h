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

	int s1;
	int s2;
private:

};


