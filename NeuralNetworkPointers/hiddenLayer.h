#pragma once

#include "Layer.h"
#include <vector>
#include "Neuron.h"

using namespace std;

class Layer;
class Neuron;

class hiddenLayer : public Layer
{
public:
	hiddenLayer();
	~hiddenLayer();

	void initialiseHiddenLayer();

private:

};


