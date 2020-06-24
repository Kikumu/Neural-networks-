#include <vector>
#include "outputLayer.h"
#pragma once


using namespace std;
class Convolve
{
public:
	Convolve();
	~Convolve();
	void convole1(double[][100],int);
	vector<vector<double>>featureMapData1;
	vector<vector<double>>featureMapData2;
	vector<vector<double>>featureMapData3; //final feature map
	void convolve2(vector<double>,int,int);
	void convolve3(vector<double>,int,int);
	int counter1;

	//for back prop(last layer only)
	vector<vector<double>>data1; //weight data
	vector<vector<double>>data2;//weight data (size 12 2 by 2)
	vector<vector<double>>data3;//weight data(size 24 2 by 2)

	//to be used in conjuction with data for back propagation.
	vector<double>filter_summary;
	vector<double>filter_summary1;
	vector<double>filter_summary2;
	vector<double>filter_summary3;

	//vector<vector<double>>inputbackprop1; //for last layer
	void backpropagation();
	
	vector<double>Flattened_features;

	void flatten(int);

	

	int datacounter1 = NULL;
	int datacounter2 = NULL;
	//double** convole1(double[][100]);
private:

};

