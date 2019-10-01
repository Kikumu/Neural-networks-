#include "Layer.h"
#include <random>
#include<ctime>
#include <vector>
#include "Eigen/Core"
#include "Training.h"

Training traintype;
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

void Layer::forwardPropagate(double** i)
{
	//create a vector of weights to multiply activation map with; since conv is 60 by 60
	Eigen::Matrix<double, 60, 1>weights;
	Eigen::Matrix<double, 60, 60>activationMap;
	Eigen::Matrix<double, 60, 1>layerValues; //aka neurons

	if (Firstweight.size() < 1) {
		mt19937 generator;
		generator.seed(time(0));
		uniform_real_distribution<double>hue(0, 1);
		double random = hue(generator);
		for (int i = 0; i < 60; i++) //vector of 60
		{
			weights(i, 0) = (random = hue(generator)) / 10000;
			Firstweight.push_back(random = hue(generator));
		}
	}
	
	for (int r = 0; r < 60; r++)
	{
		for (int c = 0; c < 60; c++) {
			activationMap(r,c) = i[r][c];
		}
	}
	double vals = NULL;

	//propagate
	layerValues = activationMap * weights;
	
	for (int i = 0; i < 60; i++)
	{
		//will optimize later
		vals = layerValues(i, 0);
		layerValues(i, 0) = traintype.funcSwish(vals);
		firstLayerData.push_back(layerValues(i, 0));
	}
	//firstLayerData.push_back(layerData);
	//double** propagateData = NULL;
}

void Layer::forwardPropagate2(double** i)
{
	Eigen::Matrix<double, 2, 2>activationMap;
	Eigen::Matrix<double, 2, 1>weights;
	Eigen::Matrix<double, 2, 1>layerValues; //aka neurons

	if (SecondWeight.size() < 1) {
		//create a vector of weights to multiply activation map with; since conv is 60 by 60
		mt19937 generator;
		generator.seed(time(0));
		uniform_real_distribution<double>hue(0, 1);
		double random = hue(generator);

		//weights initialised(should only be done once)
		//double* weightData = NULL;
		//weightData = new double[60];

		for (int i = 0; i < 2; i++) //vector of 60
		{
			weights(i, 0) = (random = hue(generator)) / 10000;
			SecondWeight.push_back(random = hue(generator));
		}
	}
	
	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			activationMap(r, c) = i[r][c];
		}
	}

	double vals = NULL;
	//propagate
	layerValues = activationMap * weights;
	//apply swish

	for (int i = 0; i < 2; i++)
	{
		//will optimize later
		vals = layerValues(i, 0);
		layerValues(i, 0) = traintype.funcSwish(vals);
		secondLayerData.push_back(layerValues(i, 0)); //output
	}
}

double Layer::LayerSensitivity()
{
	return 0.0;
}



void Layer::costRes(double totalNumberOfTrainingInputs, double inputs, double predicted_output[2], double desired_output[2])
{
	Eigen::Vector2d actual_output;
	Eigen::Vector2d network_output;
	Eigen::Vector2d difference;
	double length;
	double costData1;


	for (int i = 0; i < 2; i++) {
		network_output(i) = predicted_output[i]; //copied layer output to eigen
		actual_output(i) = desired_output[i];
	}

	//difference = actual_output - network_output;
	difference = network_output - actual_output;
	length = difference.squaredNorm();
	double sum = 0;
	for (int i = 0; i < inputs; i++)
	{
		sum += length;
	}
	costData1 = (1.0 / (2.0 * inputs)) * sum;
	costData.push_back(costData1);
}

//void Layer::costResDer(double totalNumberOfTrainingInputs, double inputs, double predicted_output[2], double desired_output[2])
//{
//	Eigen::Vector2d actual_output;
//	Eigen::Vector2d network_output;
//	Eigen::Vector2d difference;
//	double length;
//	double costData1;
//
//
//	for (int i = 0; i < 2; i++) {
//		network_output(i) = predicted_output[i]; //copied layer output to eigen
//		actual_output(i) = desired_output[i];
//	}
//
//	difference = actual_output - network_output;
//	length = difference.norm()*2;
//	double sum = 0;
//	for (int i = 0; i < inputs; i++)
//	{
//		sum += length;
//	}
//	costData1 = (1.0 / (2.0 * inputs)) * sum;
//	costData.push_back(costData1);
//}

void Layer::init(double mu, double sigma)
{
}

void Layer::backpropagation()
{
	double LearningRate = 0.1;
}


