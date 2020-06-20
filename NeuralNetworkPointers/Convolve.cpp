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
	else {
			saveFilter1 = data1.at(0);
			saveFilter2 = data1.at(1);
			saveFilter3 = data1.at(2);
			saveFilter4 = data1.at(3);
			int k = 0;

			for (int r = 0; r < 5; r++)
			{
				for (int c = 0; c < 5; c++) {
					FilterSize(r, c) = saveFilter1.at(k);
					++k;
				}
			}
			
			k = 0;
			for (int r = 0; r < 5; r++)
			{
				for (int c = 0; c < 5; c++) {
					FilterSize1(r, c) = saveFilter2.at(k);
					++k;
				}
			}
			
			k = 0;
			for (int r = 0; r < 5; r++)
			{
				for (int c = 0; c < 5; c++) {
					FilterSize2(r, c) = saveFilter3.at(k);
					++k;
				}
			}
			
			k = 0;
			for (int r = 0; r < 5; r++)
			{
				for (int c = 0; c < 5; c++) {
					FilterSize3(r, c) = saveFilter4.at(k);
					++k;
				}
			}
			k = 0;
			
	}
	

	//feature 1
	for (int r = 0; r < 100; r++) {
		if (r < 95) {
			r += stride;
			for (int c = 0; c < 100; c++) {
				if (c < 95) {
					c += stride;
					inputChunk = input.block(r, c, 5, 5);
					pre_activation = inputChunk * FilterSize;
					for (int r = 0; r < 5; r++) {
						for (int c = 0; c < 5; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}

	featureMapData1.push_back(filter_summary);
	//feature 2
	for (int r = 0; r < 100; r++) {
		if (r < 95) {
			r += stride;
			for (int c = 0; c < 100; c++) {
				if (c < 95) {
					c += stride;
					inputChunk = input.block(r, c, 5, 5);
					pre_activation = inputChunk * FilterSize1;
					for (int r = 0; r < 5; r++) {
						for (int c = 0; c < 5; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary1.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData1.push_back(filter_summary1);
	//feature 3
	for (int r = 0; r < 100; r++) {
		if (r < 95) {
			r += stride;
			for (int c = 0; c < 100; c++) {
				if (c < 95) {
					c += stride;
					inputChunk = input.block(r, c, 5, 5);
					pre_activation = inputChunk * FilterSize2;
					for (int r = 0; r < 5; r++) {
						for (int c = 0; c < 5; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary2.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData1.push_back(filter_summary2);
	//feature 4
	for (int r = 0; r < 100; r++) {
		if (r < 95) {
			r += stride;
			for (int c = 0; c < 100; c++) {
				if (c < 95) {
					c += stride;
					inputChunk = input.block(r, c, 5, 5);
					pre_activation = inputChunk * FilterSize3;
					for (int r = 0; r < 5; r++) {
						for (int c = 0; c < 5; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary3.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData1.push_back(filter_summary3);
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

	if (data2.size() != 12) {
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
	else 
	{
		
		saveFilter1 = data2.at(0);
		saveFilter2 = data2.at(1) ;
		saveFilter3 = data2.at(2);
		saveFilter4 = data2.at(3);
		





		int k = 0;

		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize(r, c) = saveFilter1.at(k);
				++k;
			}
		}

		k = 0;
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize1(r, c) = saveFilter2.at(k);
				++k;
			}
		}

		k = 0;
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize2(r, c) = saveFilter3.at(k);
				++k;
			}
		}

		k = 0;
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize3(r, c) = saveFilter4.at(k);
				++k;
			}
		}
		k = 0;

	}
	
	//f1
	for (int r = 0; r < 15; r++) {
		if (r < 13) {
			//r += stride;
			for (int c = 0; c < 15; c++) {
				if (c < 13) {
					//c += stride;
					inputChunk = input.block(r, c, 2, 2);
					pre_activation = inputChunk * FilterSize;
					for (int r = 0; r < 2; r++) {
						for (int c = 0; c < 2; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData2.push_back(filter_summary);

	//f2
	for (int r = 0; r < 15; r++) {
		if (r < 13) {
			//r += stride;
			for (int c = 0; c < 15; c++) {
				if (c < 13) {
					//c += stride;
					inputChunk = input.block(r, c, 2, 2);
					pre_activation = inputChunk * FilterSize1;
					for (int r = 0; r < 2; r++) {
						for (int c = 0; c < 2; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary1.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData2.push_back(filter_summary1);

	//f3
	for (int r = 0; r < 15; r++) {
		if (r < 13) {
			//r += stride;
			for (int c = 0; c < 15; c++) {
				if (c < 13) {
					//c += stride;
					inputChunk = input.block(r, c, 2, 2);
					pre_activation = inputChunk * FilterSize2;
					for (int r = 0; r < 2; r++) {
						for (int c = 0; c < 2; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary2.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData2.push_back(filter_summary2);

	//f4
	for (int r = 0; r < 15; r++) {
		if (r < 13) {
			//r += stride;
			for (int c = 0; c < 15; c++) {
				if (c < 13) {
					//c += stride;
					inputChunk = input.block(r, c, 2, 2);
					pre_activation = inputChunk * FilterSize3;
					for (int r = 0; r < 2; r++) {
						for (int c = 0; c < 2; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary3.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData2.push_back(filter_summary3);
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
	vector<double>ic_tkr;
	vector<double>ic_tkr1;
	Eigen::Matrix<double, 8, 8>input;
	Eigen::Matrix<double, 2, 2>pre_activation;

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

	if (data3.size() != 24) {
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
	else {
		//24. loop through filter one and 2 separately
		
		saveFilter1 = data3.at(0);
		
		saveFilter2 = data3.at(1);
		
		int k = 0;

		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize(r, c) = saveFilter1.at(k);
				++k;
			}
		}

		k = 0;
		for (int r = 0; r < 2; r++)
		{
			for (int c = 0; c < 2; c++) {
				FilterSize1(r, c) = saveFilter2.at(k);
				++k;
			}
		}

		k = 0;

	}
	

	//f1
	for (int r = 0; r < 8; r++) {
		if (r < 6) {
			r += stride;
			for (int c = 0; c < 8; c++) {
				if (c < 6) {
					c += stride;
					inputChunk = input.block(r, c, 2, 2);
					//int z = 0;
					for (int i = 0; i < 2; i++) {
						for (int j = 0; j < 2; j++)
							ic_tkr.push_back(inputChunk(i, j)); ////////////in
					}
					pre_activation = inputChunk * FilterSize;
					for (int r = 0; r < 2; r++) {
						for (int c = 0; c < 2; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData3.push_back(filter_summary);
	inputbackprop1.push_back(ic_tkr);
	//f2
	for (int r = 0; r < 8; r++) {
		if (r < 6) {
			r += stride;
			for (int c = 0; c < 8; c++) {
				if (c < 6) {
					c += stride;
					inputChunk = input.block(r, c, 2, 2);
					pre_activation = inputChunk * FilterSize1;
					for (int i = 0; i < 2; i++) {
						for (int j = 0; j < 2; j++)
							ic_tkr1.push_back(inputChunk(i, j)); ////////////
					}
					for (int r = 0; r < 2; r++) {
						for (int c = 0; c < 2; c++) {
							activation_data = t.funcSwish(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary1.push_back(sum);
				}
				sum = 0.0;
			}
		}
	}
	featureMapData3.push_back(filter_summary);
	inputbackprop1.push_back(ic_tkr1);
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

