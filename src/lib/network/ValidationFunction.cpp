#include "ValidationFunction.h"

#include <cmath>
#include <cassert>

using namespace neural::details;
using namespace arma;


void ValidationFunction::eltMult(arma::vec const & mult_p, arma::vec & out_p) const
{
	assert(mult_p.n_rows==out_p.n_rows);
	for ( size_t i(0) ; i < out_p.n_rows ; ++ i ) {
		out_p.at(i) *= mult_p.at(i);
	}
}

void ValidationFunction::apply(arma::vec const & sum_p, arma::vec & res_p) const
{
	assert(sum_p.n_rows==res_p.n_rows);
	for ( size_t i(0) ; i < res_p.n_rows ; ++ i ) {
		res_p.at(i) = apply(sum_p.at(i));
	}
}
void ValidationFunction::reverse(arma::vec const & sum_p, arma::vec & res_p) const
{
	assert(sum_p.n_rows==res_p.n_rows);
	for ( size_t i(0) ; i < res_p.n_rows ; ++ i ) {
		res_p.at(i) = reverse(sum_p.at(i));
	}
}

float ValidationFunction::apply(float const & x) const
{
	return 1/(1+std::exp(-x));
}
float ValidationFunction::reverse(float const & x) const
{
	float intermediate_l(1+std::exp(x));
	return std::exp(x)/(intermediate_l*intermediate_l);
}