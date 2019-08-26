
#include <iostream>
#include <iomanip>
#include "inputLayer.h"
#include "weights.h"
#include "hiddenLayer.h"

vector<weights*>inputWeightsTemp;
vector<Neuron*>TempNeurons;
vector<Neuron*>hLayerTemp;
int  NumberOfInputNeurons = NULL;
int  NumberOfHiddenLayerNeurons = NULL;
int  numberOfHiddenLayers = NULL;
vector<hiddenLayer>hLayer;

using namespace std;
//TODO: CREATE BIAS
int main(int, char**) {
	inputLayer input;
	hiddenLayer hidden;

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
	cin >> numberOfHiddenLayers;

	vector<Neuron*>hiddenLayerNeurons(NumberOfHiddenLayerNeurons); //set number of neurons in hidden layer
	//we need number of hidden layer neurons because of neuron to neuron connection
	//the number of weights in input neuron is dependent on the number of neurons in the neuron layer
	for (int i = 0; i < NumberOfInputNeurons; i++) {
		Neuron* iNeuron = new Neuron;
		for (int j = 0; j < NumberOfHiddenLayerNeurons; j++) {
			weights* iWeight = new weights;
			Neuron* hNeuron = new Neuron; //hiddenLayerNeuron
			iWeight->SetOneNeuron(iNeuron);
			iWeight->setTwoNeuron(hiddenLayerNeurons.at(j));
			iWeight->setWeight(iNeuron->randomizeWeights());
			inputWeightsTemp.push_back(iWeight);
			iNeuron->setWeightsOut(inputWeightsTemp);
			//error here
			//hiddenLayerNeurons.at(j)->addWeightIn(iWeight);
			//hLayerTemp.push_back(hNeuron);
		}
		inputWeightsTemp.clear();
		TempNeurons.push_back(iNeuron);
	}
	input.setNumberOfNeurons(TempNeurons);
	TempNeurons.clear();

	//setup hidden layer
	for (int i = 0; i < NumberOfHiddenLayerNeurons; i++)
	{
		//for first layer, the incoming weights==outgoing input weights
		Neuron* hNeuron = new Neuron;
		vector<weights*>tmp;
		Neuron* tmpN = new Neuron;
		//vector<Neuron*>tempN;
		//first hidden layer weights == number of neurons in input layer
		//NB: TWO BECOMES ONE AND ONE BECOMES TWO YA DIG
		for (int j = 0; j < NumberOfInputNeurons; j++) {
			weights * hOutWeight = new weights; //FOR THE OUTGOING WEIGHT INITIALIZER
			hOutWeight->SetOneNeuron(hNeuron);
			hOutWeight->setWeight(hNeuron->randomizeWeights());
			//FOR INCOMING WEIGHTS
			//ACCEPT ALL WEIGHT NEURON FROM EACH NEURON
			//i just want the first weights of each neuron for the first round
			//each neuron is gonna get the same 
			tmpN = input.getNeurons().at(j);
			//tmpN.
			tmp.push_back(tmpN->getWeightsOut().at(i));//stores weight from input lyr
			
		}
	}
	system("PAUSE");
}