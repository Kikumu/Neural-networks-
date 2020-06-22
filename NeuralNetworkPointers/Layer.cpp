#include "Layer.h"
#include <random>
#include<ctime>
#include <vector>
#include "Eigen/Core"
#include "Training.h"


Training traintype;
CostFunction cost;
using namespace std;


Layer::Layer(int i, int h):number_of_inputs_size(i),number_of_output_size(h) {
	number_of_inputs_size = NULL;
	number_of_output_size = NULL;
}

Layer::Layer()
{
	number_of_inputs_size = NULL;
	number_of_output_size = NULL;
}

Layer::~Layer()
{

}

int Layer::getInputSize()
{
	return number_of_inputs_size;
}

int Layer::getOutputSize()
{
	return number_of_output_size;
}

void Layer::forwardPropagate(vector<double>i)
{
	//create a vector of weights to multiply activation map with; since conv is 60 by 60
	Eigen::Matrix<double, 1280, 1>weights;
	Eigen::Matrix<double, 128, 1>activationMap; //Data from flattened layer
	
	double dot; //dot product per push

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);
	if (Firstweight.size() < 1) {
		//weight initialization set to 1280. (cause 10 neurons in layer since flatenned layer contains 128 values)
			for (int i = 0; i < 1280; i++) //vector of 94
			{
				weights(i, 0) = (random = hue(generator)) / 128;
				//Firstweight.push_back(random = hue(generator));
				Firstweight.push_back(weights(i, 0));
			}
	}

	//GETS INPUTS FROM CONV FLATTENED LAYER
	for (int r = 0; r < 128; r++)
	{
		activationMap(r, 0) = i[r];
	}
	double vals = NULL;

	//propagate
	int limiter = 0;
	int counter = 0;
	while (limiter < 10) {
		Eigen::Matrix<double, 128, 1>temp_weights;
		int j = 0;
		while (counter < 1280) {
			temp_weights(j, 0) = weights(counter, 0);
			if (counter == 127)
				break;
			else if (counter > 126 && counter % 127 == 0)
				break;
			counter++;
			j++;
		}
		//INCREMENT SO THAT ITS CARRIED OVER TO NEXT ITERATION
		counter++;
		//DOT PRODUCT
		dot = activationMap.dot(temp_weights);
		//ACTIVATION
		vals = traintype.fncSigmoid(dot);
		firstLayerData.push_back(vals);
		limiter++;
	}
}

void Layer::forwardPropagate2(vector<double>i)
{
	Eigen::Matrix<double, 100, 1>weights;      //holds all weight data
	Eigen::Matrix<double, 10, 1>activationMap; //input from previous data layer
	double dot;
	double vals;
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	//INPUT DATA GRAB
	for (int r = 0; r < 10; r++)
	{
		activationMap(r, 0) = i[r];
	}

	//obtain changed weights(if any)
	if (SecondWeight.size() > 1) {
		for (int j = 0; j < 100; j++) {
			weights(j, 0) = SecondWeight[j];
		}
	}

	if (SecondWeight.size() < 1) {
		//WEIGHT INITIALIZATION
		
			for (int j = 0; j < 100; j++) {
				weights(j, 0) = (random = hue(generator)) / 10; //divide b number of inputs to scale properly
				SecondWeight.push_back(weights(j, 0));
			}


	}
	
	//PROPAGATION
	int limiter = 0;
	int counter = 0;
	while (limiter < 10) {
		Eigen::Matrix<double, 10, 1>temp_weights;
		int j = 0;
		while (counter < 100) {
			temp_weights(j, 0) = weights(counter, 0);
			if (counter == 9)
				break;
			else if (counter > 9 && counter % 10 == 0)
				break;
			counter++;
			j++;
		}
		//INCREMENT SO THAT ITS CARRIED OVER TO NEXT ITERATION
		counter++;
		//DOT PRODUCT
		dot = activationMap.dot(temp_weights);
		//ACTIVATION
		vals = traintype.fncSigmoid(dot);
		secondLayerData.push_back(vals);
		limiter++;
	}
}

void Layer::forwardPropagate3(vector<double>i)
{
	Eigen::Matrix<double, 20, 1>weights;
	Eigen::Matrix<double, 10, 1>activationMap; //150 batches of these(each unique)
	//create a vector of weights to multiply activation map with; since conv is 60 by 60

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);
	//obtain changed weights(if any)
	if (ThirdWeight.size() > 1) {
		for (int j = 0; j < 20; j++) {
			weights(j, 0) = ThirdWeight[j];
		}
	}

	if (ThirdWeight.size() < 1) {
		//WEIGHT INITIALIZATION
			for (int j = 0; j < 20; j++) {
				weights(j, 0) = (random = hue(generator)) / 10;
				ThirdWeight.push_back(weights(j, 0));
			}
	}

	double vals = NULL;
	double dot = NULL;
	//PREVIOUS LAYER DATA
	for (int r = 0; r < 10; r++)
	{
		activationMap(r, 0) = i[r];
	}

	//PROPAGATION
	int limiter = 0;
	int counter = 0;
	while (limiter < 2) {
		Eigen::Matrix<double, 10, 1>temp_weights;
		int j = 0;
		while (counter < 20) {
			temp_weights(j, 0) = weights(counter, 0);
			if (counter == 9)
				break;
			else if (counter > 9 && counter % 10 == 0)
				break;
			counter++;
			j++;
		}
		//INCREMENT SO THAT ITS CARRIED OVER TO NEXT ITERATION
		counter++;
		//DOT PRODUCT
		dot = activationMap.dot(temp_weights);
		//ACTIVATION
		vals = traintype.fncSigmoid(dot);  //ACTIVATION
		ThirdWeightData.push_back(vals);
		limiter++;
	}
}

double Layer::LayerSensitivity()
{
	return 0.0;
}


void Layer::init(double mu, double sigma)
{
}

void Layer::backpropagation(double c, double s)
{
	int weights_loop = 0;
	int limiter = 0; //neuron_loop
	double cost_ = c;

	//Third_layer
	while (weights_loop < 20) {
		//limiter = 0;
		while (limiter < 10) {
			double out_data = secondLayerData[limiter]; //grab neuron information
			double associated_weight = ThirdWeight[weights_loop];
			double new_out_data;
			//weight update
			new_out_data = traintype.fncSigmoidDerivative(out_data);
			double new_weight;
			double temp = learning_rate * new_out_data * s * c;
			new_weight = associated_weight + temp;
			ThirdWeight[weights_loop] = new_weight; //update info
			limiter++;
			break;
		}
		if (limiter > 8 && limiter % 9 == 0)
			limiter = 0;
		weights_loop++;
	}

	weights_loop = 0;
	limiter = 0;
	//Second Layer
	while (weights_loop < 100) {
		//limiter = 0;
		while (limiter < 10) {
			double out_data = firstLayerData[limiter]; //grab neuron information(current dat layer)
			double out_data1 = secondLayerData[limiter]; // grab neuron info(prev dat layer)
			double associated_weight = SecondWeight[weights_loop];
			double new_out_data;
			//weight update
			new_out_data = traintype.fncSigmoidDerivative(out_data);//take current derivative
			double new_weight;
			double temp = learning_rate * new_out_data * out_data1; //currdev/prev dev/cost dev
			new_weight = associated_weight + temp;
			SecondWeight[weights_loop] = new_weight; //update info
			limiter++;
			break;
		}
		if (limiter > 8 && limiter % 9 == 0)
			limiter = 0;
		weights_loop++;
	}



		
}


