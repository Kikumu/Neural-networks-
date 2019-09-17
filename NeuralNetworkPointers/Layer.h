#pragma once

#include <vector>
#include "Eigen/Core"

using namespace std;
using namespace Eigen;

//class Neuron;
//class weights;
//placeholder for neurons
class Layer
{
public:
	/*Layer(const int i, const int h) :number_of_inputs_size(i), number_of_output_size(h) {
	};*/
	Layer(int i, int h);

	~Layer();

	typedef Eigen::Matrix<double, Dynamic, Dynamic>Matrix; //calculated
	typedef Eigen::Matrix<double, Dynamic, 1>Vector; //value passed onto layer

	int getInputSize();
	int getOutputSize();

	void forwardPropagate(Matrix);

	

	virtual void backpropagation();
protected:
	int number_of_inputs_size; //input rows
	int number_of_output_size; //layer rows


private:
	virtual void init(double mu, double sigma); //weights
};


