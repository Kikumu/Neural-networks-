#include "Matrix.h"

Matrix::Matrix()
{
}

Matrix::~Matrix()
{
}

float & Matrix::operator()(int row, int col)
{
	return data[row + col * rows]; //nswtd. apparently some way to acces values in matrices?
}

void Matrix::multiply(Matrix &a, Matrix &b) //addresses
{
	if (a.rows != b.columns) {
		throw "matrix cannot be multiplied";
	}
	if (this== &a || this== &b) {
		throw "result matrix contains same value as passed parametres";
	}
	int sharedDimensions = b.columns;//or a.columns cause they share same dimensions to multiply together
	for (int i = 0; i != a.rows; i++) {

		for (int j = 0; j != b.columns; j++) {

			for (int k = 0; k < sharedDimensions; k++) {
				total += a(i, k) *b(k, j); //wikipedia
			}
			this->operator()(i, j) = total;
		}
	}
}

void Matrix::hadamard(Matrix & a)
{
	if (this->rows != a.rows || this->columns != a.columns) {
		throw "ERROR";
	}
	for (int i = 0; i < this->total; i++) {
		data[i] *= a.data[i]; //change this star here to add multiply etc etc
	}
}
