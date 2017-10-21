#include "Network.h"

#include "Layer.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

using namespace neural;
using namespace arma;

template<typename T>
struct deleteFunctor
{
	void operator()(T& obj)
	{
		delete obj;
	}
};

Network::Network(size_t depth_p, size_t width_p, size_t inputSize_p, size_t outputSize_p)
	: _depth(depth_p+2)
	, _width(width_p)
	, _inputSize(inputSize_p)
	, _outputSize(outputSize_p)
	, _layers(depth_p+2, NULL)
{
	if ( outputSize_p > width_p ) {
		throw std::logic_error("Cannot create a network narrower than solution");
	}
	// Create layers from output to input
	// output
	_layers[_depth-1] = new details::Layer(_outputSize, _validator);
	for ( size_t i(_depth-2) ; i > 0 ; -- i )
	{
		_layers[i] = new details::Layer(width_p, _validator, _layers[i+1]);
	}
	// input
	_layers[0] = new details::Layer(_inputSize, _validator, _layers[1]);
}
Network::~Network()
{
	std::for_each(_layers.begin(), _layers.end(), deleteFunctor<details::Layer *>());
}

void Network::forward(std::vector<double> const & input_p)
{
	// set up input
	assert(input_p.size()==_inputSize);
	for ( size_t i(0) ; i < input_p.size() ; ++ i )
	{
		_layers[0]->setRes(i,input_p[i]);
	}
	// move forward
	for ( size_t i(0) ; i < _depth-1 ; ++ i )
	{
		_layers[i]->forward();
	}
}
void Network::backward(std::vector<double> const & expectedOutput_p)
{
	// set up output
	details::Layer * outputLayer_l(_layers[_depth-1]);
	assert(expectedOutput_p.size()==_outputSize);
	for ( size_t i(0) ; i < expectedOutput_p.size() ; ++ i )
	{
		double diff(expectedOutput_p[i]-outputLayer_l->getRes(i));
		double sprime(_validator.reverse(outputLayer_l->getSum(i)));
		/*std::cout<<"Delta output sum = "<<"S'("<<outputLayer_l->getSum(i)<<") * "<<diff<<std::endl;
		std::cout<<"Delta output sum = "<<sprime<<" * "<<diff<<std::endl;*/
		outputLayer_l->setDSum(i,diff * sprime);
	}
	// move backward
	for ( int i(_depth-2) ; i >= 0 ; -- i )
	{
		_layers[i]->backward();
	}
}

void Network::train(std::vector<details::TrainingData> const & trainingData_p, size_t iterMax_p)
{
	if ( trainingData_p.size() == 0 || iterMax_p == 0 )
	{
		return;
	}
	// transition matrices update
	std::vector<mat *> dTransition_l;
	dTransition_l.push_back(new mat(_width, _inputSize+1, fill::zeros));
	for ( size_t i(1) ; i < _depth-2 ; ++ i )
	{
		dTransition_l.push_back(new mat(_width, _width+1, fill::zeros));
	}
	dTransition_l.push_back(new mat(_outputSize, _width+1, fill::zeros));

	// training coef
	double alpha = 1.0;

	for ( size_t i(0) ; i < iterMax_p ; ++ i )
	{
		// Reset matrices
		if ( i > 0 )
		{
			for ( size_t i(0) ; i < _depth-1 ; ++ i )
			{
				dTransition_l[i]->zeros();
			}
		}
		for ( size_t n(0) ; n < trainingData_p.size() ; ++ n )
		{
			forward(trainingData_p[n].input);
			backward(trainingData_p[n].expectedOutput);

			// matrices update
			for ( size_t k(0) ; k < _depth-1 ; ++ k )
			{
				(*dTransition_l[k]) += getLayer(k)->getDTransition();
			}
		}

		// Normalize matrices update and apply
		for ( size_t k(0) ; k < _depth-1 ; ++ k )
		{
			(*dTransition_l[k]) *= alpha/trainingData_p.size();
			getLayer(k)->updateTransition(*dTransition_l[k]);
		}
	}
	// Delete matrices update
	std::for_each(dTransition_l.begin(), dTransition_l.end(), deleteFunctor<mat *>());
}

std::vector<double> Network::getOutput() const
{
	std::vector<double> vect_l;
	for ( size_t i(0) ; i < _outputSize ; ++ i )
	{
		vect_l.push_back(_layers[_depth-1]->getRes(i));
	}
	return vect_l;
}
std::vector<double> Network::getInput() const
{
	std::vector<double> vect_l;
	for ( size_t i(0) ; i < _inputSize ; ++ i )
	{
		vect_l.push_back(_layers[0]->getRes(i));
	}
	return vect_l;
}
/**
 * Layer 0 is input, Layer _depth-1 is output
 */
details::Layer * Network::getLayer(size_t depth_p)
{
	assert(depth_p<_depth);
	return _layers[depth_p];
}