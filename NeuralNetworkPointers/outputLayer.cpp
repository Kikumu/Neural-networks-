#include "outputLayer.h"
#include "Eigen\Core"
#include <random>
#include<ctime>
#include "Training.h"

Training trainOut;

//find a way to accumilate allweights

outputLayer::outputLayer()
{
}

outputLayer::~outputLayer()
{
}

void outputLayer::calc()
{
	//calculate each weight with feature map, each weight must me unique?
	vector<double>FeatureData;//for looping through features
	vector<double>Outputdata; //should only be 2 values
	Eigen::Matrix<double, 2, 2>FeatureTensor; //each feature data will be put in a feature tensor and multiplied by a certain weight filter
	Eigen::Matrix<double, 2, 2>WeightTensor;
	Eigen::Matrix<double, 2, 2>Pre_;
	double dat = 0.0;
	double sum = 0.0;
	//store weight tensors..after i finish the output
	//48 WEIGHT TENSORS, YIKES
	//WELL 48*4
	//finally, summation

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	int k = 0;
	for (int i = 0; i < 24; i++) {
		//features[i] = FeatureData;
		FeatureData = features[i];
		for (int r = 0; r < 2; r++) {
			for (int c = 0; c < 2; c++) {
				FeatureTensor(r, c) = FeatureData[k];
				++k;
			}
		}
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				WeightTensor(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
			}
		}
		//multiply feature tensor by a weight and get output value
		Pre_ = WeightTensor * FeatureTensor;
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				dat = trainOut.fncSigmoid(Pre_(r, c));
				sum += dat;
				Pre_(r, c) = dat;
			}
		}
		k = 0;
	}
	s1 = sum;
}

void outputLayer::calc1()
{
	//calculate each weight with feature map, each weight must me unique?
	vector<double>FeatureData;//for looping through features
	vector<double>Outputdata; //should only be 2 values
	Eigen::Matrix<double, 2, 2>FeatureTensor; //each feature data will be put in a feature tensor and multiplied by a certain weight filter
	Eigen::Matrix<double, 2, 2>WeightTensor;
	Eigen::Matrix<double, 2, 2>Pre_;
	double dat = 0.0;
	double sum = 0.0;
	//store weight tensors..after i finish the output
	//48 WEIGHT TENSORS, YIKES
	//WELL 48*4
	//finally, summation

	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	int k = 0;
	for (int i = 0; i < 24; i++) {
		//features[i] = FeatureData;
		FeatureData = features[i];
		for (int r = 0; r < 2; r++) {
			for (int c = 0; c < 2; c++) {
				FeatureTensor(r, c) = FeatureData[k];
				++k;
			}
		}
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				WeightTensor(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
			}
		}
		//multiply feature tensor by a weight and get output value
		Pre_ = WeightTensor * FeatureTensor;
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				dat = trainOut.fncSigmoid(Pre_(r, c));
				sum += dat;
				Pre_(r, c) = dat;
			}
		}
		k = 0;
	}
	s2 = sum;
}
