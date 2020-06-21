#include "Convolve.h"
#include "Eigen/Core"
#include <random>
#include<ctime>
#include "Training.h"
#include <iostream>

using namespace std;
Training t;
Convolve::Convolve()
{
}

Convolve::~Convolve()
{
}

void Convolve::convole1(double i[][100])
{
	//outputsize calc = (inputwidth - filterwidth)/stride + 1
	int stride = 4;
	double sum = 0.0;
	double activation_data;
	vector<double>filter_summary; //save this to each filter map
	vector<double>filter_summary1;
	vector<double>filter_summary2;
	vector<double>filter_summary3;
	Eigen::Matrix<double, 5, 5>FilterSize; //weights
	Eigen::Matrix<double, 5, 5>FilterSize1; //weights
	Eigen::Matrix<double, 5, 5>FilterSize2; //weights
	Eigen::Matrix<double, 5, 5>FilterSize3; //weights
	Eigen::Matrix<double, 5, 5>inputChunk;
	Eigen::Matrix<double, 100, 100>input;
	Eigen::Matrix<double, 5, 5>pre_activation;

	for (int r = 0; r < 100; r++) {
		for (int c = 0; c < 100; c++) {
			input(r, c) = i[r][c];
		}
	}
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);
	vector<double>saveFilter1;
	vector<double>saveFilter2;
	vector<double>saveFilter3;
	vector<double>saveFilter4;

	if (data1.size() == 0)
	{
		//weights initialisation
		for (int r = 0; r < 5; r++)
		{
			for (int c = 0; c < 5; c++) {
				FilterSize(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
				saveFilter1.push_back(FilterSize(r, c));
			}
		}
		data1.push_back(saveFilter1);
		for (int r = 0; r < 5; r++)
		{
			for (int c = 0; c < 5; c++) {
				FilterSize1(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
				saveFilter2.push_back(FilterSize1(r, c));
			}
		}
		data1.push_back(saveFilter2);
		for (int r = 0; r < 5; r++)
		{
			for (int c = 0; c < 5; c++) {
				FilterSize2(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
				saveFilter3.push_back(FilterSize2(r, c));
			}
		}
		data1.push_back(saveFilter3);
		for (int r = 0; r < 5; r++)
		{
			for (int c = 0; c < 5; c++) {
				FilterSize3(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
				saveFilter4.push_back(FilterSize3(r, c));
			}
		}
		data1.push_back(saveFilter4);
	}
	//forward propagation
	int feature_counter = 0;
	while (feature_counter < 4) {
		for (int r = 0; r < 100; r++) {
			if (r < 95) { //set to 95 due to filter size
				r += stride;
				for (int c = 0; c < 100; c++) {
					if (c < 95) {
						c += stride;
						inputChunk = input.block(r, c, 5, 5);
						saveFilter1 = data1.at(feature_counter);
						int k = 0;
						for (int r = 0; r < 5; r++)
						{
							for (int c = 0; c < 5; c++) {
								FilterSize(r, c) = saveFilter1.at(k);
								++k;
							}
						}
						k = 0;
						pre_activation = inputChunk * FilterSize;
						for (int r = 0; r < 5; r++) {
							for (int c = 0; c < 5; c++) {
								//activation for forward propagation
								activation_data = t.funcSwish(pre_activation(r, c));
								sum += activation_data;
							}
						}
						if (feature_counter == 0) 
							filter_summary.push_back(sum);
						else if (feature_counter == 1)
							filter_summary1.push_back(sum);
						else if (feature_counter == 2)
							filter_summary2.push_back(sum);
						else if (feature_counter == 3) 
							filter_summary3.push_back(sum);
					}
					sum = 0.0;
				}
			}
		}
		if(feature_counter == 0)featureMapData1.push_back(filter_summary);
		else if(feature_counter == 1)featureMapData1.push_back(filter_summary1);
		else if (feature_counter == 2)featureMapData1.push_back(filter_summary2);
		else if (feature_counter == 3)featureMapData1.push_back(filter_summary3);
		feature_counter++;
	}
	
}

void Convolve::convolve2(vector<double>in)
{
	//outputsize calc = (inputwidth - filterwidth)/stride + 1
	int stride = 0;
	double sum = 0.0;
	double activation_data = 0.0;
	
	Eigen::Matrix<double, 2, 2>FilterSize; //weights
	Eigen::Matrix<double, 2, 2>FilterSize1; //weights
	Eigen::Matrix<double, 2, 2>FilterSize2; //weights
	Eigen::Matrix<double, 2, 2>FilterSize3; //weights
	Eigen::Matrix<double, 2, 2>inputChunk;
	Eigen::Matrix<double, 15, 15>input;
	Eigen::Matrix<double, 2, 2>pre_activation;

	//CONVERT INPUT VECTOR BACK TO MATRIX
	int k = 0;
	for (int r = 0; r < 15; r++) {
		for (int c = 0; c < 15; c++) {
			input(r, c) = in[k];
			++k;
		}
	}
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	vector<double>saveFilter1;
	vector<double>saveFilter2;
	vector<double>saveFilter3;
	vector<double>saveFilter4;

	if (data2.size() == 0) {
		//WEIGHT INITIALIZATION
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
				saveFilter1.push_back(FilterSize(r, c));
			}
		}
		data2.push_back(saveFilter1);
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize1(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
				saveFilter2.push_back(FilterSize1(r, c));
			}
		}
		data2.push_back(saveFilter2);
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize2(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
				saveFilter3.push_back(FilterSize2(r, c));
			}
		}
		data2.push_back(saveFilter3);
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize3(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
				saveFilter4.push_back(FilterSize3(r, c));
			}
		}
		data2.push_back(saveFilter4);
	}
	//FORWARD PROPAGATION

	int feature_counter = 0;
	while (feature_counter < 4) {
		//SET TO 15 TO MATCH INPUT SIZE
		for (int r = 0; r < 15; r++) {
			//SET TO 13 FOR THRESHHOLDING AND AVOIDING MATRIX SIZE MISMATCHES
			if (r < 13) {
				//r += stride;
				for (int c = 0; c < 15; c++) {
					if (c < 13) {
						//c += stride;
						inputChunk = input.block(r, c, 2, 2);
						saveFilter1 = data2.at(feature_counter);
						int k = 0;
						for (int r = 0; r < 2; r++)
						{
							for (int c = 0; c < 2; c++) {
								FilterSize(r, c) = saveFilter1.at(k);
								++k;
							}
						}
						k = 0;
						pre_activation = inputChunk * FilterSize;
						for (int r = 0; r < 2; r++) {
							for (int c = 0; c < 2; c++) {
								activation_data = t.funcSwish(pre_activation(r, c)); //ACTIVATION FUNCTION(SEE TRAINING.CPP FOR MORE OPTIONS)
								sum += activation_data;
							}
						}
						if (feature_counter == 0)
							filter_summary.push_back(sum);
						else if (feature_counter == 1)
							filter_summary1.push_back(sum);
						else if (feature_counter == 2)
							filter_summary2.push_back(sum);
						else if (feature_counter == 3)
							filter_summary3.push_back(sum);
					}
					sum = 0.0;
				}
			}
		}
		if (feature_counter == 0)featureMapData2.push_back(filter_summary);
		else if (feature_counter == 1)featureMapData2.push_back(filter_summary1);
		else if (feature_counter == 2)featureMapData2.push_back(filter_summary2);
		else if (feature_counter == 3)featureMapData2.push_back(filter_summary3);
		feature_counter++;
	}
}

void Convolve::convolve3(vector<double>in)
{
	//outputsize calc = (inputwidth - filterwidth)/stride + 1
	int stride = 2;
	double sum = 0.0;
	double activation_data = 0.0;
	vector<double>filter_summary; //save this to each filter map
	vector<double>filter_summary1; //save this to each filter map
	Eigen::Matrix<double, 2, 2>FilterSize; //weights
	Eigen::Matrix<double, 2, 2>FilterSize1; //weights
	Eigen::Matrix<double, 2, 2>inputChunk;
	Eigen::Matrix<double, 8, 8>input;
	Eigen::Matrix<double, 2, 2>pre_activation;
	//INPUT VECTOR CONVERTED TO MATRIX
	int k = 0;
	for (int r = 0; r < 8; r++) {
		for (int c = 0; c < 8; c++) {
			input(r, c) = in[k];
			++k;
		}
	}
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	vector<double>saveFilter1;
	vector<double>saveFilter2;

	if (data3.size() == 0) {
		//WEIGHT INITIALIZATION
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
				saveFilter1.push_back(FilterSize(r, c));
			}
		}
		data3.push_back(saveFilter1);
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize1(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
				saveFilter2.push_back(FilterSize1(r, c));
			}
		}
		data3.push_back(saveFilter2);
	}

	//32(16*2). loop through filter one and 2 separately
	//FORWARD PROPAGATION
	int feature_counter = 0;
	while (feature_counter < 2) {
		for (int r = 0; r < 8; r++) {
			if (r < 6) {
				r += stride;
				for (int c = 0; c < 8; c++) {
					if (c < 6) {
						c += stride;
						inputChunk = input.block(r, c, 2, 2);
						saveFilter1 = data3.at(feature_counter);
						int k = 0;
						for (int r = 0; r < 2; r++)
						{
							for (int c = 0; c < 2; c++) {
								FilterSize(r, c) = saveFilter1.at(k);
								++k;
							}
						}
						k = 0;
						pre_activation = inputChunk * FilterSize;
						for (int r = 0; r < 2; r++) {
							for (int c = 0; c < 2; c++) {
								activation_data = t.funcSwish(pre_activation(r, c));
								sum += activation_data;
							}
						}
						if (feature_counter == 0)
							filter_summary.push_back(sum);
						else if (feature_counter == 1)
							filter_summary1.push_back(sum);
					}
					sum = 0.0;
				}
			}
		}
		if (feature_counter == 0)featureMapData3.push_back(filter_summary);
		else if (feature_counter == 1)featureMapData3.push_back(filter_summary1);
		feature_counter++;
	}
}

void Convolve::flatten()
{
		vector<double>itr;
	for (int i = 0; i < featureMapData3.size(); i++)
	{
		itr = featureMapData3[i]; //save ech value in a vector
		for (int j = 0; j < itr.size(); j++) {
			Flattened_features.push_back(itr.at(j));
		}
	}
}

