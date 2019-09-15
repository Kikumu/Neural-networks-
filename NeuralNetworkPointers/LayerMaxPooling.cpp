#include "LayerMaxPooling.h"
#include "Eigen/Core"



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
	array2d = new double* [100];//create rows

	for (int r = 0; r < 10; r++) {
		array2d[r] = new double[100]; //create cols
		for (int c = 0; c < 10; c++) {
			array2d[r][c] = 200;
		}
	}

	return array2d;
}

double** LayerMaxPooling::resultant2(int a, double b[][100])
{
	double** array2d = NULL;
	Eigen::Matrix<double, 100, 100>sizer;
	for (int i = 0; i != 5; i++) {
		for (int j = 0; j != 5; j++) {
		   sizer(i,j) = rand()%10;
		}
	}
	//lets say stride of 3 right in a 5 by 5 matrix
	//need to store all blocks obtain max values, delete blocks
	//sizer.block
	
	
	for (int i = 0; i < 3; i++) {
		//need to account for row and column
		for (int j = 0; j < 3; j++) {
			sizer.block(i, j, 3, 3);
		}
	}

	return nullptr;
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




