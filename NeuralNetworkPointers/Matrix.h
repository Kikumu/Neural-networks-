#pragma once


class Matrix
{
public:
	Matrix();
	~Matrix();
	float* data;
	int rows;
	int columns;
	int total;
	float& operator()(int row, int col);

	void multiply(Matrix&, Matrix&);
	void hadamard(Matrix&);
private:
};

