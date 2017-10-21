
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


	std::vector<std::vector<float> > vecInput_l(4, std::vector<float>());
	vecInput_l[0].push_back(2.f);
	vecInput_l[0].push_back(2.f);
	vecInput_l[1].push_back(1.f);
	vecInput_l[1].push_back(2.f);
	vecInput_l[2].push_back(2.f);
	vecInput_l[2].push_back(1.f);
	vecInput_l[3].push_back(1.f);
	vecInput_l[3].push_back(1.f);

	std::vector<std::vector<float> > vecExpected_l(4, std::vector<float>());
	vecExpected_l[0].push_back(0.f);
	vecExpected_l[1].push_back(1.f);
	vecExpected_l[2].push_back(1.f);
	vecExpected_l[3].push_back(0.f);

	size_t maxIter_l(200000);
	for ( size_t i(0) ; i < maxIter_l ; ++ i )
	{
		network_l.forward(vecInput_l[i%4]);

		/*std::cout<<"Sum"<<std::endl;

		std::cout	<<layerHidden_l->getSum(0)<<";"
					<<layerHidden_l->getSum(1)<<";"
					<<layerHidden_l->getSum(2)<<std::endl;

		std::cout	<<layerOutput_l->getSum(0)<<std::endl;*/

		/*std::cout<<"Res"<<std::endl;

		std::cout	<<layerHidden_l->getRes(0)<<";"
					<<layerHidden_l->getRes(1)<<";"
					<<layerHidden_l->getRes(2)<<std::endl;*/


		std::cout	<<layerOutput_l->getRes(0)<<std::endl;

		network_l.backward(vecExpected_l[i%4]);

		/*std::cout<<"dSum"<<std::endl;

		std::cout	<<layerInput_l->getDSum(0)<<";"
					<<layerInput_l->getDSum(1)<<std::endl;

		std::cout	<<layerHidden_l->getDSum(0)<<";"
					<<layerHidden_l->getDSum(1)<<";"
					<<layerHidden_l->getDSum(2)<<std::endl;

		std::cout	<<layerOutput_l->getDSum(0)<<std::endl;

		std::cout<<"new weight"<<std::endl;
		std::cout<<layerInput_l->getWeight(0,0)<<std::endl;
		std::cout<<layerInput_l->getWeight(0,1)<<std::endl;
		std::cout<<layerInput_l->getWeight(0,2)<<std::endl;
		std::cout<<layerInput_l->getWeight(1,0)<<std::endl;
		std::cout<<layerInput_l->getWeight(1,1)<<std::endl;
		std::cout<<layerInput_l->getWeight(1,2)<<std::endl;
		std::cout<<layerHidden_l->getWeight(0,0)<<std::endl;
		std::cout<<layerHidden_l->getWeight(1,0)<<std::endl;
		std::cout<<layerHidden_l->getWeight(2,0)<<std::endl;*/
	}
	
/*
	vec a = randu<mat>(2);
	vec b = randu<mat>(2);
	vec d = randu<mat>(2);; d.fill(1.0);
	mat c = a * (d.t() / b.t());

	a.print("a:");
	b.print("b:");
	c.print("c:");*/

	return 0;
}
