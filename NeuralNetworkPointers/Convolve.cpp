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

void Convolve::convole1(double i[][100], int chk)
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
	//for(chk)
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
								activation_data = t.fncSigmoid(pre_activation(r, c));
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
		if (feature_counter == 0 && chk == 0) {
			featureMapData1.push_back(filter_summary);
			filter_summary.clear();
		}
		else if (feature_counter == 0 && chk != 0) {
			featureMapData1[feature_counter] = filter_summary;
			filter_summary.clear();
		}


		else if (feature_counter == 1 && chk == 0) {
			featureMapData1.push_back(filter_summary1);
			filter_summary1.clear();
		}
		else if (feature_counter == 1 && chk != 0) {
			featureMapData1[feature_counter] = filter_summary1;
			filter_summary1.clear();
		}



		else if (feature_counter == 2 && chk == 0){
			featureMapData1.push_back(filter_summary2);
			filter_summary2.clear();
		}
		else if (feature_counter == 2 && chk != 0) {
			featureMapData1[feature_counter] = filter_summary2;
			filter_summary2.clear();
		}




		else if (feature_counter == 3 && chk == 0) {
			featureMapData1.push_back(filter_summary3);
			filter_summary3.clear();
		}
		else if (feature_counter == 3 and chk != 0) {
			featureMapData1[feature_counter] = filter_summary3;
			filter_summary3.clear();
		}
		feature_counter++;
	}
}





void Convolve::convolve2(vector<double>in,int chk, int epch)
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

	int feature_counter = chk;
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
								activation_data = t.fncSigmoid(pre_activation(r, c)); //ACTIVATION FUNCTION(SEE TRAINING.CPP FOR MORE OPTIONS)
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
		if (feature_counter == 0 && epch == 0) {
			featureMapData2.push_back(filter_summary);
			filter_summary.clear();
		}
		else if (feature_counter == 0 && epch != 0) {
			featureMapData2[feature_counter] = filter_summary;
			filter_summary.clear();
		}


		else if (feature_counter == 1 && epch == 0) {
			featureMapData2.push_back(filter_summary1);
			filter_summary1.clear();
		}
		else if (feature_counter == 1 && epch != 0) {
			featureMapData2[feature_counter] = filter_summary1;
			filter_summary1.clear();
		}



		else if (feature_counter == 2 && epch == 0) {
			featureMapData2.push_back(filter_summary2);
			filter_summary2.clear();
		}
		else if (feature_counter == 2 && epch != 0) {
			featureMapData2[feature_counter] = filter_summary2;
			filter_summary2.clear();
		}




		else if (feature_counter == 3 && epch == 0) {
			featureMapData2.push_back(filter_summary3);
			filter_summary3.clear();
		}
		else if (feature_counter == 3 and epch != 0) {
			featureMapData2[feature_counter] = filter_summary3;
			filter_summary3.clear();
		}
		//feature_counter++;
		break;
	}
}

void Convolve::convolve3(vector<double>in, int chk, int epch)
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
	int feature_counter = chk;
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
								activation_data = t.fncSigmoid(pre_activation(r, c));
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
		if (feature_counter == 0 && epch == 0) {
			featureMapData3.push_back(filter_summary);
			filter_summary.clear();
		}
		else if (feature_counter == 0 && epch != 0) {
			featureMapData3[feature_counter] = filter_summary;
			filter_summary.clear();
		}


		else if (feature_counter == 1 && epch == 0) {
			featureMapData3.push_back(filter_summary1);
			filter_summary1.clear();
		}
		else if (feature_counter == 1 && epch != 0) {
			featureMapData3[feature_counter] = filter_summary1;
			filter_summary1.clear();
		}

		break;
	}
}

void Convolve::flatten(int epch)
{
	vector<double>itr;
	if (epch == 0) {
		for (int i = 0; i < featureMapData3.size(); i++)
		{
			itr = featureMapData3[i]; //save ech value in a vector
			for (int j = 0; j < itr.size(); j++) {
				Flattened_features.push_back(itr.at(j));
			}
		}
	}
	else
	{
		for (int i = 0; i < featureMapData3.size(); i++)
		{
			itr = featureMapData3[i]; //save ech value in a vector
			for (int j = 0; j < itr.size(); j++) {
				Flattened_features[j] = (itr.at(j));
			}
		}
	}
	
}

void Convolve::backpropagation(vector<vector<double>> f2, vector<vector<double>> f1)
{
	double learning_rate = 0.5;

	//FIRST CONVOLUTION/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Eigen::Matrix<double, 2, 2>FilterSize; //weights
	Eigen::Matrix<double, 2, 2>FilterSize1; //weights
	vector<double>temp;
	//feature map 3(past)
	//feature map 2(current)
	//weights of fm 2[i] = curr[i] + lr*der(fm3)*der()
	int flattened_loop = 0;
	int filter_loop = 0; //iterates through filter(2 - 2 by 2)
	double new_out_data_1 = 0;
	double out_data_1;

	while (flattened_loop < 8) {

		//first feature filter update
		while (flattened_loop < 4) {
			out_data_1 = Flattened_features[flattened_loop];
			new_out_data_1 +=  t.fncSigmoidDerivative(out_data_1);
			break;
			//new_out_data_1 = (1.0 / (25.0 * t.fncSigmoidDerivative(out_data_1)));
		}

		//so....we flip the kernel..by 180...and then work on it "normally" while changing feature weights
		//WEIGHT ONLY BACK PROPAGATED TO THE "WINNING" NEURON BECAUSE OF MAXPOOLING.
		//SINCE ITS A 5 BY 5 CONVOLUTION IT WILL BE OUTPUT DERIVATIVE MULTIPLIED BY 1/25
		if (flattened_loop > 2 && flattened_loop == 3) {
			new_out_data_1 = (1.0 / (25.0 * new_out_data_1));
			//update to filter(in data 3)
			//i want to obtain data from data3(0)
			temp = data3[0];
			for (int r = 0; r < 4; r++)
			{
				temp[r] = temp[r] + (learning_rate * new_out_data_1);
			}
			data3[0] = temp;
			new_out_data_1 = 0.0;
		}

		while (flattened_loop > 3)
		{
			out_data_1 = Flattened_features[flattened_loop];
			new_out_data_1 += t.fncSigmoidDerivative(out_data_1);
			break;
		}
		if (flattened_loop > 6 && flattened_loop == 7) {
			new_out_data_1 = (1.0 / (25.0 * new_out_data_1));
			//update to filter(in data 3)
			//i want to obtain data from data3(0)
			temp = data3[1];
			for (int r = 0; r < 4; r++)
			{
				temp[r] = temp[r] + (learning_rate * new_out_data_1);
			}
			data3[1] = temp;
			new_out_data_1 = 0.0;
		}

		flattened_loop++;

	}


	//flattened_loop = 0;
	//filter_loop = 0; //iterates through filter(2 - 2 by 2)
	//new_out_data_1 = 0;
	//out_data_1 = 0;
	//double out_data = f[limiter]; //grab neuron information(current dat layer(128))
	//double out_data1 = firstLayerData[data_loop];// grab neuron info(prev dat layer)
	//double associated_weight = Firstweight[weights_loop];
	////firstweight, first layer data and flattened conv layer(f) 
	//double new_out_data;
	//double new_out_data_1;
	////weight update
	//new_out_data = t.fncSigmoidDerivative(out_data);//take current derivative

	//SECOND CONVOLUTION///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//8 by 8 (64)
	// 4 filters
	//(64 summations and pass back error)
	
	for (int feature_loop = 0; feature_loop < f2.size(); feature_loop++) {
		vector<double>temp = f2[feature_loop]; //temp incoming data storage
		vector<double>weights_temp = data2[feature_loop]; //weights hold temp
		//find summation/derivatives
		double sum = 0;
		for (int data_loop = 0; data_loop < temp.size(); data_loop++) {
			sum += t.fncSigmoidDerivative(temp[data_loop]);
		} 
		sum = (1.0 / (25.0 * sum));
		//weight change loop
		for (int weights_loop = 0; weights_loop < data2[feature_loop].size(); weights_loop++) {
			weights_temp[weights_loop] = weights_temp[weights_loop] + learning_rate * sum; //update
		}
		data2[feature_loop] = weights_temp;
		sum = 0;
	}



	for (int feature_loop = 0; feature_loop < f1.size(); feature_loop++) {
		vector<double>temp = f1[feature_loop]; //temp incoming data storage
		vector<double>weights_temp = data1[feature_loop]; //weights hold temp
		//find summation/derivatives
		double sum = 0;
		for (int data_loop = 0; data_loop < temp.size(); data_loop++) {
			sum += t.fncSigmoidDerivative(temp[data_loop]);
		}
		sum = (1.0 / (25.0 * sum));
		//weight change loop
		for (int weights_loop = 0; weights_loop < data1[feature_loop].size(); weights_loop++) {
			weights_temp[weights_loop] = weights_temp[weights_loop] + learning_rate * sum; //update
		}
		data1[feature_loop] = weights_temp;
		sum = 0;
	}

}


