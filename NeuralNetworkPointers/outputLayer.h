#pragma once
#include "Layer.h"
#include <vector>

class outputLayer
{
public:
	outputLayer();
	~outputLayer();

	//each feature will have its own weight

	vector<vector<double>>features; //use this for backprop
	//void calc();
	//void calc1();

	double s1;
	double s2;


	//for back prop
	//vector<vector<double>>outputBackpropData;
private:

};


