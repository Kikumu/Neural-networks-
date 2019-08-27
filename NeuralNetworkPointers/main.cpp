
#include <iostream>
#include <iomanip>
#include "inputLayer.h"
#include "weights.h"
#include "hiddenLayer.h"
#include "Neuron.h"
#include "outputLayer.h"

vector<weights*>inputWeightsTemp;
vector<weights*>hiddenWeightsTemp;
vector<Neuron*>TempNeurons;
vector<Neuron*>hLayerTemp;
int  NumberOfInputNeurons = NULL;
int  NumberOfHiddenLayerNeurons = NULL;
int  numberOfOutputNeurons = NULL;
vector<hiddenLayer>hLayer;
vector<Neuron*>tempHlayerNeurons;

//int size = NULL;
//NB: lets first work with a single layered network shall we

using namespace std;
//TODO: CREATE BIAS
int main(int, char**) {
	inputLayer input;
	hiddenLayer hidden;
	outputLayer out;
	//Neuron* const iNeuron = new Neuron;
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
	cout << "Set number of hidden layers";
	cout << "\n";
	cin >> numberOfOutputNeurons;
	cout << "\n";
	cout << "****Setting up....****";
	//vector<Neuron*>hiddenLayerNeurons(NumberOfHiddenLayerNeurons); //set number of neurons in hidden layer
	vector<Neuron*>hiddenLayerNeurons;
	//we need number of hidden layer neurons because of neuron to neuron connection
	//the number of weights in input neuron is dependent on the number of neurons in the neuron layer
	for (int i = 0; i < NumberOfInputNeurons; i++) {
		Neuron* iNeuron = new Neuron;
		for (int j = 0; j != NumberOfHiddenLayerNeurons; ++j) {
			weights* iWeight = new weights;
			Neuron* hNeuron = new Neuron; //hiddenLayerNeuron
			iWeight->SetOneNeuron(iNeuron);
			iWeight->setTwoNeuron(hNeuron);
			hNeuron->addWeightsIn(iWeight);
			iWeight->setWeight(iNeuron->randomizeWeights());
			iNeuron->addWeightsOut(iWeight);
			int size = hiddenLayerNeurons.size();
			if(size < NumberOfHiddenLayerNeurons)
			hiddenLayerNeurons.push_back(hNeuron);
			else 
			{
				hiddenLayerNeurons.at(j)->addWeightsIn(iWeight);
			}
			///if(size >= hiddenLayerNeurons.size())
		}
		//hiddenLayerNeurons.push_back(hNeuron);
		//inputWeightsTemp.clear();
		TempNeurons.push_back(iNeuron);
	}
	input.setNumberOfNeurons(TempNeurons);
	TempNeurons.clear();
	hidden.setNumberOfNeurons(hiddenLayerNeurons);
	system("PAUSE");
}