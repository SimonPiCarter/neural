
#include "hello.h"

#include <iostream>

#include <armadillo>

#include "network/Network.h"
#include "network/Layer.h"
#include "network/ValidationFunction.h"

using namespace arma;
using namespace neural;

int main()
{
	std::cout<<test::hello("Simon")<<std::endl;

	/*size_t dimX = 5;
	size_t dimY = 2;
	mat A = randu<mat>(dimX,dimY);
	vec b = randu<mat>(dimX);

	vec x;
	A.print("A:");
	b.print("b:");

	bool ok = solve(x,A,b, solve_opts::no_approx);
	std::cout<<"ok:"<<ok<<std::endl;
	std::cout<<"x:"<<x.n_rows<<";"<<x.n_cols<<std::endl;
	std::cout<<"A:"<<A.n_rows<<";"<<A.n_cols<<std::endl;
	std::cout<<"b:"<<b.n_rows<<";"<<b.n_cols<<std::endl;

	x.print("x:");
	A.print("A:");
	pinv(A).print("A.inv:");

	vec z = A * x;
	z.print("z:");

	vec res(dimX);*/

	Network network_l(1,3,2,1);

	details::Layer * layerInput_l = network_l.getLayer(0);
	details::Layer * layerHidden_l = network_l.getLayer(1);
	details::Layer * layerOutput_l = network_l.getLayer(2);

	layerInput_l->setWeight(0,0,0.8f);
	layerInput_l->setWeight(0,1,0.4f);
	layerInput_l->setWeight(0,2,0.3f);
	layerInput_l->setWeight(1,0,0.2f);
	layerInput_l->setWeight(1,1,0.9f);
	layerInput_l->setWeight(1,2,0.5f);
	layerHidden_l->setWeight(0,0,0.3f);
	layerHidden_l->setWeight(1,0,0.5f);
	layerHidden_l->setWeight(2,0,0.9f);

	std::vector<details::TrainingData> trainingData_l;
	trainingData_l.push_back(details::TrainingData());
	trainingData_l.back().input.push_back(1.0);
	trainingData_l.back().input.push_back(1.0);
	trainingData_l.back().expectedOutput.push_back(0.0);
	trainingData_l.push_back(details::TrainingData());
	trainingData_l.back().input.push_back(1.0);
	trainingData_l.back().input.push_back(0.0);
	trainingData_l.back().expectedOutput.push_back(1.0);
	trainingData_l.push_back(details::TrainingData());
	trainingData_l.back().input.push_back(0.0);
	trainingData_l.back().input.push_back(1.0);
	trainingData_l.back().expectedOutput.push_back(1.0);
	trainingData_l.push_back(details::TrainingData());
	trainingData_l.back().input.push_back(0.0);
	trainingData_l.back().input.push_back(0.0);
	trainingData_l.back().expectedOutput.push_back(0.0);

	network_l.train(trainingData_l);

	for ( size_t k(0) ; k < trainingData_l.size() ; ++ k )
	{
		network_l.forward(trainingData_l[k].input);
		std::vector<double> output_l = network_l.getOutput();
		std::cout<<"output : "<<std::endl;
		for ( size_t i(0) ; i < output_l.size() ; ++ i )
		{
			std::cout<<output_l[i];
			if ( i < output_l.size()-1)
			{
				std::cout<<", ";
			} else {
				std::cout<<std::endl;
			}
		}
		std::cout<<"expected : "<<std::endl;
		for ( size_t i(0) ; i < trainingData_l[k].expectedOutput.size() ; ++ i )
		{
			std::cout<<trainingData_l[k].expectedOutput[i];
			if ( i < trainingData_l[k].expectedOutput.size()-1)
			{
				std::cout<<", ";
			} else {
				std::cout<<std::endl;
			}
		}
	}

	return 0;
}
