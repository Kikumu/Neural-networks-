#include "Flatten.h"
#include "Eigen\Core"
#include <vector>

using namespace std;
Flatten::Flatten()
{
}

Flatten::~Flatten()
{
}

void Flatten::flattener()
{
	vector<double>itr;
	for (int i = 0; i < flattenFeatures.size(); i++)
	{
		flattenFeatures[i] = itr;
	}
}
