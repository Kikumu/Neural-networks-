#pragma once
#include "Layer.h"
#include <vector>
#include "Neuron.h"

using namespace std;

class Layer;
class Neuron;

//placeholder for neurons
class inputLayer : public Layer
{
public:
	inputLayer ();
	~inputLayer();

	void initialiseInputLayer(inputLayer input);
private:
	
};





