#include "Convolve.h"
#include "Eigen/Core"
#include <random>
#include<ctime>
#include "Training.h"

using namespace std;
Training t;
Convolve::Convolve()
{
}

Convolve::~Convolve()
{
}

vector<vector<double>> Convolve::convole1(double i[][100])
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
	vector<vector<double>>featureMapData;
	//Eigen::Matrix<double, 19, 19>activationMap; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap1; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap2; //feature map
	//Eigen::Matrix<double, 19, 19>activationMap3; //feature map
	Eigen::Matrix<double, 100, 100>input;
	Eigen::Matrix<double, 5, 5>pre_activation;
	Eigen::Matrix<double, 25, 1>StretchedFilter;

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

	FilterSize1 = 2 * FilterSize1;
	FilterSize2 = 3 * FilterSize2;
	FilterSize3 = 4 * FilterSize3;

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

	featureMapData.push_back(filter_summary);
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
	featureMapData.push_back(filter_summary1);
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
	featureMapData.push_back(filter_summary2);
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
	featureMapData.push_back(filter_summary3);
	return featureMapData;
}

//double** Convolve::convole1(double i[][100])
//{
//	//outputsize calc = (inputwidth - filterwidth)/stride + 1
//	int stride = 4;
//	double sum = 0.0;
//	double activation_data;
//	vector<double>filter_summary;
//	vector<double>filter_summary1;
//	vector<double>filter_summary2;
//	vector<double>filter_summary3;
//	Eigen::Matrix<double, 5, 5>FilterSize; //weights
//	Eigen::Matrix<double, 5, 5>FilterSize1; //weights
//	Eigen::Matrix<double, 5, 5>FilterSize2; //weights
//	Eigen::Matrix<double, 5, 5>FilterSize3; //weights
//	Eigen::Matrix<double, 5, 5>inputChunk;
//	Eigen::Matrix<double, 19, 19>activationMap; //feature map
//	Eigen::Matrix<double, 19, 19>activationMap1; //feature map
//	Eigen::Matrix<double, 19, 19>activationMap2; //feature map
//	Eigen::Matrix<double, 19, 19>activationMap3; //feature map
//	Eigen::Matrix<double, 100, 100>input;
//	Eigen::Matrix<double, 5, 5>pre_activation;
//	Eigen::Matrix<double, 25, 1>StretchedFilter;
//
//	for (int r = 0; r < 100; r++) {
//		for (int c = 0; c < 100; c++) {
//			input(r, c) = i[r][c];
//		}
//	}
//	mt19937 generator;
//	generator.seed(time(0));
//	uniform_real_distribution<double>hue(0, 1);
//	double random = hue(generator);
//
//	for (int r = 0; r < 5; r++)
//	{
//		for (int c = 0; c < 5; c++) {
//			FilterSize(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
//		}
//	}
//	for (int r = 0; r < 5; r++)
//	{
//		for (int c = 0; c < 5; c++) {
//			FilterSize1(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
//		}
//	}
//	for (int r = 0; r < 5; r++)
//	{
//		for (int c = 0; c < 5; c++) {
//			FilterSize2(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
//		}
//	}
//	for (int r = 0; r < 5; r++)
//	{
//		for (int c = 0; c < 5; c++) {
//			FilterSize3(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
//		}
//	}
//
//	FilterSize1 = 2 * FilterSize1;
//	FilterSize2 = 3 * FilterSize2;
//	FilterSize3 = 4 * FilterSize3;
//
//	//feature 1
//	for (int r = 0; r < 100; r++) {
//		if (r < 95) {
//			r += stride;
//			for (int c = 0; c < 100; c++) {
//				if (c < 95) {
//					c += stride;
//					inputChunk = input.block(r, c, 5, 5);
//					pre_activation = inputChunk*FilterSize;
//					for (int r = 0; r < 5; r++) {
//						for (int c = 0; c < 5; c++) {
//							activation_data = t.funcSwish(pre_activation(r, c));
//							sum += activation_data;
//						}
//					}
//					filter_summary.push_back(sum);
//				}
//				sum = 0.0;
//			}
//		}
//	}
//   //feature 2
//	for (int r = 0; r < 100; r++) {
//		if (r < 95) {
//			r += stride;
//			for (int c = 0; c < 100; c++) {
//				if (c < 95) {
//					c += stride;
//					inputChunk = input.block(r, c, 5, 5);
//					pre_activation = inputChunk * FilterSize1;
//					for (int r = 0; r < 5; r++) {
//						for (int c = 0; c < 5; c++) {
//							activation_data = t.funcSwish(pre_activation(r, c));
//							sum += activation_data;
//						}
//					}
//					filter_summary1.push_back(sum);
//				}
//				sum = 0.0;
//			}
//		}
//	}
//	//feature 3
//	for (int r = 0; r < 100; r++) {
//		if (r < 95) {
//			r += stride;
//			for (int c = 0; c < 100; c++) {
//				if (c < 95) {
//					c += stride;
//					inputChunk = input.block(r, c, 5, 5);
//					pre_activation = inputChunk * FilterSize2;
//					for (int r = 0; r < 5; r++) {
//						for (int c = 0; c < 5; c++) {
//							activation_data = t.funcSwish(pre_activation(r, c));
//							sum += activation_data;
//						}
//					}
//					filter_summary2.push_back(sum);
//				}
//				sum = 0.0;
//			}
//		}
//	}
//	//feature 4
//	for (int r = 0; r < 100; r++) {
//		if (r < 95) {
//			r += stride;
//			for (int c = 0; c < 100; c++) {
//				if (c < 95) {
//					c += stride;
//					inputChunk = input.block(r, c, 5, 5);
//					pre_activation = inputChunk * FilterSize3;
//					for (int r = 0; r < 5; r++) {
//						for (int c = 0; c < 5; c++) {
//							activation_data = t.funcSwish(pre_activation(r, c));
//							sum += activation_data;
//						}
//					}
//					filter_summary3.push_back(sum);
//				}
//				sum = 0.0;
//			}
//		}
//	}
//
//
//	return nullptr;
//}
