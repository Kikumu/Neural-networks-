#pragma once

#include "Layer.h"
#include "Convolve.h"
#include "Training.h"

//monitor weights
//update weights accordingly as per error/cost data
//Save weights to a text file
//error of each neuron
using namespace std;
class BackPropagation
{
public:
	BackPropagation();
	~BackPropagation();

	//updated layer data
	vector<vector<double>>updated_data1; //weight data
	vector<vector<double>>updated_data2;//weight data
	vector<vector<double>>updated_data3;//weight data

	//layer update functions
	void update_layer1(double learning_rate, vector<vector<double>>updated_data1);
	void update_layer2(double learning_rate, vector<vector<double>>updated_data1);
	void update_layer3(double learning_rate, vector<vector<double>>updated_data1);

private:

};

