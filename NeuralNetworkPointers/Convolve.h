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

	vector<vector<double>>inputbackprop1; //for last layer
	



	int datacounter1 = NULL;
	int datacounter2 = NULL;
	//double** convole1(double[][100]);
private:

};

