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
	//Eigen::Matrix<double, 19, 19>activationMap; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap1; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap2; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap3; //feature map
	Eigen::Matrix<double, 100, 100>input;
	Eigen::Matrix<double, 5, 5>pre_activation;
	//Eigen::Matrix<double, 25, 1>StretchedFilter;

	for (int r = 0; r < 100; r++) {
		for (int c = 0; c < 100; c++) {
			input(r, c) = i[r][c];
		}
	}
	mt19937 generator;
	generator.seed(time(0));
	uniform_real_distribution<double>hue(0, 1);
	double random = hue(generator);

	for (int r = 0; r < 5; r++)
	{
		for (int c = 0; c < 5; c++) {
			FilterSize(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
		}
	}
	for (int r = 0; r < 5; r++)
	{
		for (int c = 0; c < 5; c++) {
			FilterSize1(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
		}
	}
	for (int r = 0; r < 5; r++)
	{
		for (int c = 0; c < 5; c++) {
			FilterSize2(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
		}
	}
	for (int r = 0; r < 5; r++)
	{
		for (int c = 0; c < 5; c++) {
			FilterSize3(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
		}
	}

	/*FilterSize1 = 2 * FilterSize1;
	FilterSize2 = 3 * FilterSize2;
	FilterSize3 = 4 * FilterSize3;*/

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
	vector<double>filter_summary; //save this to each filter map
	vector<double>filter_summary1;
	vector<double>filter_summary2;
	vector<double>filter_summary3;
	Eigen::Matrix<double, 2, 2>FilterSize; //weights
	Eigen::Matrix<double, 2, 2>FilterSize1; //weights
	Eigen::Matrix<double, 2, 2>FilterSize2; //weights
	Eigen::Matrix<double, 2, 2>FilterSize3; //weights
	Eigen::Matrix<double, 2, 2>inputChunk;
	//Eigen::Matrix<double, 19, 19>activationMap; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap1; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap2; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap3; //feature map
	Eigen::Matrix<double, 15, 15>input;
	Eigen::Matrix<double, 2, 2>pre_activation;
	//Eigen::Matrix<double, 25, 1>StretchedFilter;

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

	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			FilterSize(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
		}
	}
	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			FilterSize1(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
		}
	}
	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			FilterSize2(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
		}
	}
	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			FilterSize3(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
		}
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
	//Eigen::Matrix<double, 19, 19>activationMap; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap1; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap2; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap3; //feature map
	Eigen::Matrix<double, 8, 8>input;
	Eigen::Matrix<double, 2, 2>pre_activation;
	//Eigen::Matrix<double, 25, 1>StretchedFilter;

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

	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			FilterSize(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
		}
	}

	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 2; c++) {
			FilterSize1(r, c) = (random = hue(generator)) / 4; //divide by 5 to scale with number of inputs
		}
	}

	//f1
	for (int r = 0; r < 8; r++) {
		if (r < 6) {
			r += stride;
			for (int c = 0; c < 8; c++) {
				if (c < 6) {
					c += stride;
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
	featureMapData3.push_back(filter_summary);

	//f2
	for (int r = 0; r < 8; r++) {
		if (r < 6) {
			r += stride;
			for (int c = 0; c < 8; c++) {
				if (c < 6) {
					c += stride;
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
	featureMapData3.push_back(filter_summary);
}

