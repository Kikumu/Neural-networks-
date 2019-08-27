
#include <iostream>
#include <iomanip>
#include "inputLayer.h"
#include "weights.h"
#include "hiddenLayer.h"
#include "Neuron.h"
#include "outputLayer.h"
#include "BuildNetwork.h"

int  NumberOfInputNeurons = NULL;
int  NumberOfHiddenLayerNeurons = NULL;
int  numberOfOutputNeurons = NULL;
vector<Neuron*>TempNeurons; //used in input
vector<Neuron*>outputLayerNeurons;
vector<Neuron*>hiddenLayerNeurons;

//NB: lets first work with a single layered network shall we

using namespace std;
//TODO: CREATE BIAS
int main(int, char**) {
	inputLayer input;
	hiddenLayer hidden;
	outputLayer out;
	BuildNetwork network;
	//lets say i want 4 constant neurons in my inputlayer
	//meaning store 4 constant pointers in my inputneurons
	//need to create an empty set of desired neurons and push them into the inputlayer
	cout << "set number of neurons in input Layer";
	cout << "\n";
	cin >> NumberOfInputNeurons;
	cout << "****DONE****";
	cout << "\n";
	cout << "Input number of HiddenLayer Neurons";
	cout << "\n";
	cin >> NumberOfHiddenLayerNeurons;
	cout << "****DONE****";
	cout << "\n";
	cout << "Set number of output layer neurons";
	cout << "\n";
	cin >> numberOfOutputNeurons;
	cout << "\n";
	cout << "****Setting up....****";

	//vector<Neuron*>hiddenLayerNeurons(NumberOfHiddenLayerNeurons); //set number of neurons in hidden layer

	//we need number of hidden layer neurons because of neuron to neuron connection
	//the number of weights in input neuron is dependent on the number of neurons in the neuron layer
	for (int i = 0; i < NumberOfInputNeurons; i++) {
		Neuron* iNeuron = new Neuron;
		for (int j = 0; j != NumberOfHiddenLayerNeurons; ++j) {
			weights* iWeight = new weights;
			//weights* hWeight = new weights;
			Neuron* hNeuron = new Neuron; //hiddenLayerNeuron
			iWeight->SetOneNeuron(iNeuron);
			iWeight->setTwoNeuron(hNeuron);
			hNeuron->addWeightsIn(iWeight);
			iWeight->setWeight(iNeuron->randomizeWeights());
			iNeuron->addWeightsOut(iWeight);
			//neuron limiter
			int size = hiddenLayerNeurons.size(); //limiter for hidden neurons
			if (size < NumberOfHiddenLayerNeurons)
			{
				//no need to set a new weight for output neuron it only takes in weights
				for (int k = 0; k != numberOfOutputNeurons; k++) {
					Neuron* oNeuron = new Neuron;
					weights* hWeight = new weights;
					hWeight->setWeight(hNeuron->randomizeWeights());
					hWeight->SetOneNeuron(hNeuron);
					hWeight->setTwoNeuron(oNeuron);
					oNeuron->addWeightsIn(hWeight); //point toward the same thing
					hNeuron->addWeightsOut(hWeight);//point towards the same thing
					int size1 = outputLayerNeurons.size(); //limiter for output neurons
					if (size1 < numberOfOutputNeurons)
						outputLayerNeurons.push_back(oNeuron);
					else
						outputLayerNeurons.at(k)->addWeightsIn(hWeight);
				}
				hiddenLayerNeurons.push_back(hNeuron);
			}
			else 
			{
			//LEARNT: YOU WILL NEED TO COMMUNICATE WITH THEM DIRECTLY LIKE THIS
			hiddenLayerNeurons.at(j)->addWeightsIn(iWeight);
			}
		}
		//hiddenLayerNeurons.push_back(hNeuron);
		//inputWeightsTemp.clear();
		TempNeurons.push_back(iNeuron);
	}
	input.setNumberOfNeurons(TempNeurons);
	hidden.setNumberOfNeurons(hiddenLayerNeurons);
	out.setNumberOfNeurons(outputLayerNeurons);
	TempNeurons.clear();

	cout << "\n";
	cout << "****DONE INITIALISING****";
	cout << "\n";
	cout << "****APPLYING CALCULATIONS****";
	network.setup(input, hidden, out);

	system("PAUSE");
}