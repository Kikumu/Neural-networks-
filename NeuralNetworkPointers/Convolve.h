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
	//double** convole1(double[][100]);
private:

};

