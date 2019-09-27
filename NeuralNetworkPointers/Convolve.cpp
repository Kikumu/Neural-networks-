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

double** Convolve::convole1(double i[][100])
{
	//outputsize calc = (inputwidth - filterwidth)/stride + 1
	int stride = 5;
	double sum = 0.0;
	double activation_data;
	vector<double>filter_summary;
	Eigen::Matrix<double, 5, 5>FilterSize; //weights
	Eigen::Matrix<double, 5, 5>inputChunk;
	Eigen::Matrix<double, 19, 19>activationMap;
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
	
	for (int r = 0; r < 100; r++) {
		if (r < 95) {
			r + stride;
			for (int c = 0; c < 100; c++) {
				if (c < 95) {
					c + stride;
					inputChunk = input.block(r, c, 5, 5);
					double random = hue(generator);
					for (int r = 0; r < 5; r++)
					{
						for (int c = 0; c < 5; c++) {
							FilterSize(r, c) = (random = hue(generator)) / 25; //divide by 5 to scale with number of inputs
						}
					}
					pre_activation = inputChunk*FilterSize;
					for (int r = 0; r < 5; r++) {
						for (int c = 0; c < 5; c++) {
							activation_data = t.fncSigmoid(pre_activation(r, c));
							sum += activation_data;
						}
					}
					filter_summary.push_back(sum);
					//t.fncSigmoid(pre_activation);
					//.push_back(filter1.maxCoeff());
				}
			}
		}
	}

	return nullptr;
}
