#include <vector>
#include "outputLayer.h"
#pragma once


using namespace std;
class Convolve
{
public:
	Convolve();
	~Convolve();
	void convole1(double[][100]);
	vector<vector<double>>featureMapData1;
	vector<vector<double>>featureMapData2;
	vector<vector<double>>featureMapData3; //final feature map
	void convolve2(vector<double>);
	void convolve3(vector<double>);
	int counter1;

	//for back prop(last layer only)
	vector<vector<double>>data1; //weight data
	vector<vector<double>>data2;//weight data
	vector<vector<double>>data3;//weight data

	//to be used in conjuction with data for back propagation.
	vector<double>filter_summary;
	vector<double>filter_summary1;
	vector<double>filter_summary2;
	vector<double>filter_summary3;

	vector<vector<double>>inputbackprop1; //for last layer
	
	vector<double>Flattened_features;

	void flatten();

	

	int datacounter1 = NULL;
	int datacounter2 = NULL;
	//double** convole1(double[][100]);
private:

};

