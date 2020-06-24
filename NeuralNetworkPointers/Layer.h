#pragma once

#include <vector>
#include "Eigen/Core"
#include "CostFunction.h"

using namespace std;
using namespace Eigen;

class Layer
{
public:
	/*Layer(const int i, const int h) :number_of_inputs_size(i), number_of_output_size(h) {
	};*/
	Layer(int i, int h);
	Layer();
	~Layer();

	int getInputSize();
	int getOutputSize();

	int counter;
	double learning_rate;
	double LayerSensitivity();
	
	
	vector<double>outputLayerWeightData;



	//FOR FULLY CONNECTED LAYER
	void forwardPropagate(vector<double>);
	void forwardPropagate2(vector<double>);
	void forwardPropagate3(vector<double>);
	vector<double>Firstweight;
	vector<double>SecondWeight;
	vector<double>ThirdWeight;
	vector<double>firstLayerData;  //neurons
	vector<double>secondLayerData; //neurons
	vector<double>ThirdWeightData; //neurons
	vector<double>costData;

	//BACKPROPAGATION
	void backpropagation(double[2],double[2],vector<double>);

protected:
	int number_of_inputs_size; //input rows
	int number_of_output_size; //layer rows


private:
	virtual void init(double mu, double sigma); //weights
};


