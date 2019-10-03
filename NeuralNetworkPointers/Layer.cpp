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

void Layer::forwardPropagate(vector<double>i)
{
	//create a vector of weights to multiply activation map with; since conv is 60 by 60
	Eigen::Matrix<double, 94, 1>weights;
	Eigen::Matrix<double, 94, 1>activationMap; //150 batches of these(each unique)
	//Eigen::Matrix<double, 94, 1>layerValues; //aka neurons
	//Eigen::VectorXd
	double dot;

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);
	if (Firstweight.size() < 1) {
		for (int k = 0; k < 150; k++) {


			for (int i = 0; i < 94; i++) //vector of 94
			{
				weights(i, 0) = (random = hue(generator)) / 94;
				//Firstweight.push_back(random = hue(generator));
				Firstweight.push_back(weights(i, 0));
			}


			for (int r = 0; r < 94; r++)
			{
				activationMap(r, 0) = i[r];
			}
			double vals = NULL;

			//propagate
			//layerValues = activationMap * weights;
			dot = activationMap.dot(weights);
			vals = traintype.funcSwish(dot);
			firstLayerData.push_back(vals);
		}

		//double** propagateData = NULL;
	}
}

void Layer::forwardPropagate2(vector<double>i)
{
	Eigen::Matrix<double, 150, 1>weights;
	Eigen::Matrix<double, 150, 1>activationMap; //150 batches of these(each unique)
	//create a vector of weights to multiply activation map with; since conv is 60 by 60

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);
	if (SecondWeight.size() < 1) {
		for (int k = 0; k < 84; k++) //vector of 60
		{
			for (int j = 0; j < 150; j++) {
				weights(j, 0) = (random = hue(generator)) / 150;
				SecondWeight.push_back(weights(j, 0));
			}
	
			for (int r = 0; r < 150; r++)
			{
				activationMap(r, 0) = i[r];
			}

			double vals = NULL;
			double dot = NULL;
			//apply swish
			dot = activationMap.dot(weights);
			vals = traintype.funcSwish(dot);
			secondLayerData.push_back(vals);
		}
	}
}

void Layer::forwardPropagate3(vector<double>i)
{
	Eigen::Matrix<double, 84, 1>weights;
	Eigen::Matrix<double, 84, 1>activationMap; //150 batches of these(each unique)
	//create a vector of weights to multiply activation map with; since conv is 60 by 60

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);
	if (ThirdWeight.size() < 1) {
		for (int k = 0; k < 2; k++) //vector of 60
		{
			for (int j = 0; j < 84; j++) {
				weights(j, 0) = (random = hue(generator)) / 84;
				ThirdWeight.push_back(weights(j, 0));
			}

			for (int r = 0; r < 84; r++)
			{
				activationMap(r, 0) = i[r];
			}

			double vals = NULL;
			double dot = NULL;
			//apply swish
			dot = activationMap.dot(weights);
			vals = traintype.funcSwish(dot);
			ThirdWeightData.push_back(vals);
		}
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

	/*double sum = 0;
	for (int i = 0; i < inputs; i++)
	{
		sum += length;
	}
	costData1 = (1.0 / (2.0 * inputs)) * sum;
	costData.push_back(costData1);*/

	costData1 = (1.0 / (2.0 ))* length;
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


