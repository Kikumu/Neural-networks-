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
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	//weights initialised(should only be done once)
	//double* weightData = NULL;
	//weightData = new double[60];
	Eigen::Matrix<double, 60, 1>weights;
	for (int i = 0; i < 60; i++) //vector of 60
	{
		weights(i,0) = (random = hue(generator))/10000;
		Firstweight.push_back(random = hue(generator));
	}
	//double dref = NULL;
	Eigen::Matrix<double, 60, 60>activationMap;
	for (int r = 0; r < 60; r++)
	{
		for (int c = 0; c < 60; c++) {
			activationMap(r,c) = i[r][c];
		}
	}
	//training first push
	//traintype.funcSwish;
	double vals = NULL;
	Eigen::Matrix<double, 60, 1>layerValues; //aka neurons
	layerValues = activationMap * weights;
	//apply swish

	//double* layerData= new double [60];
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
	//create a vector of weights to multiply activation map with; since conv is 60 by 60
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	//weights initialised(should only be done once)
	//double* weightData = NULL;
	//weightData = new double[60];
	Eigen::Matrix<double, 2, 1>weights;
	for (int i = 0; i < 2; i++) //vector of 60
	{
		weights(i, 0) = (random = hue(generator)) / 10000;
		SecondWeight.push_back(random = hue(generator));
	}
	//double dref = NULL;
	Eigen::Matrix<double, 2, 2>activationMap;
	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			activationMap(r, c) = i[r][c];
		}
	}
	

	double vals = NULL;
	Eigen::Matrix<double, 2, 1>layerValues; //aka neurons
	layerValues = activationMap * weights;
	//apply swish

	//double* layerData= new double [60];
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



void Layer::init(double mu, double sigma)
{
}

void Layer::backpropagation()
{
}


