#include "LayerMaxPooling.h"
#include "Eigen/Core"
#include <vector>

using namespace std;

LayerMaxPooling::LayerMaxPooling()
{
}

LayerMaxPooling::~LayerMaxPooling()
{
}

Eigen::Matrix<double, Dynamic, Dynamic> LayerMaxPooling::resultant(int, Eigen::Matrix<double, Dynamic, Dynamic>a)
{
	return a;
	
}

double** LayerMaxPooling::resultant1(int b, double a[][100])
{
	//return a;
	double** array2d = NULL;
	Eigen::Matrix<double, Dynamic, Dynamic>sizer;
	//sizer = a;
	array2d = new double* [10];//create rows

	for (int r = 0; r < 10; r++) {
		array2d[r] = new double[10]; //create cols
		for (int c = 0; c < 10; c++) {
			array2d[r][c] = 200;
		}
	}
	return array2d;
}

double** LayerMaxPooling::resultant2(int a, double b[][100])
{
	double** array2d = NULL;
	Eigen::Matrix<double, 4, 4>sizer;
	for (int i = 0; i != 4; i++) {
		for (int j = 0; j != 4; j++) {
		   sizer(i,j) = rand()%10; //matrix for takin in input
		}
	}
	vector<double>number;//stores max val per filter
	Eigen::Matrix3d filter;
	for (int i = 0; i < 4; i++) {
		//need to account for row and column
		if (i < 2) {//threshholding. if padding wont fit go to next row/col
			for (int j = 0; j < 4; j++) {
				if (j < 2) {//threshholding. if padding wont fit go to next row/col
					filter = sizer.block(i, j, 3, 3); //block for picking out matrix
					number.push_back(filter.maxCoeff());//find maxval
				}
			}
		}
	}
	//convoluted data 4 by 4 grid
	array2d = new double* [2];
	for (int r = 0; r < 2; r++) {
		array2d[r] = new double[2];//for each col create a row thats what this means
		for (int c = 0; c < 2; c++) {
			array2d[r][c] = number[r];
		}
	}
	return array2d;
}


//MatrixXd LayerMaxPooling::resize(int, MatrixXd a)
//{
//	return a;
//}





//typedef Eigen::Matrix<double, Dynamic, Dynamic>Matrix;
//
//Matrix LayerMaxPooling::resultant(int, Matrix)
//{
//	return Matrix();
//}




