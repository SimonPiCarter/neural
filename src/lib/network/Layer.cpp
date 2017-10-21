#include "Layer.h"

#include "ValidationFunction.h"

#include <cassert>

using namespace neural::details;
using namespace arma;

Layer::Layer(size_t dimension_p, ValidationFunction const & validator_p)
	: _transition()
	, _dTransition()
	, _res(dimension_p)
	, _sum(dimension_p)
	, _dSum(dimension_p)
	, _dimension(dimension_p)
	, _validator(validator_p)
	, _next(NULL)
{

}
Layer::Layer(size_t dimension_p, ValidationFunction const & validator_p, Layer * next_p)
	: _transition(next_p->_dimension, dimension_p)
	, _dTransition(next_p->_dimension, dimension_p)
	, _res(dimension_p)
	, _sum(dimension_p)
	, _dSum(dimension_p)
	, _dimension(dimension_p)
	, _validator(validator_p)
	, _next(next_p)
{

}
Layer::~Layer()
{

}

void Layer::setWeight(size_t idxInNode_p, size_t idxOutNode_p, float value_p)
{
	assert(_next);
	assert(idxInNode_p < _dimension);
	assert(idxOutNode_p < _next->_dimension);
	_transition.at(idxOutNode_p, idxInNode_p) = value_p;
}
float Layer::getWeight(size_t idxInNode_p, size_t idxOutNode_p)
{
	assert(_next);
	assert(idxInNode_p < _dimension);
	assert(idxOutNode_p < _next->_dimension);
	return _transition.at(idxOutNode_p, idxInNode_p);
}

float Layer::getSum(size_t idxNode_p) const
{
	assert(idxNode_p<_dimension);
	return _sum.at(idxNode_p);
}
float Layer::getRes(size_t idxNode_p) const
{
	assert(idxNode_p<_dimension);
	return _res.at(idxNode_p);
}
float Layer::getDSum(size_t idxNode_p) const
{
	assert(idxNode_p<_dimension);
	return _dSum.at(idxNode_p);
}
void Layer::setRes(size_t idxNode_p, float value_p)
{
	assert(idxNode_p<_dimension);
	_res.at(idxNode_p) = value_p;	
}
void Layer::setDSum(size_t idxNode_p, float value_p)
{
	assert(idxNode_p<_dimension);
	_dSum.at(idxNode_p) = value_p;
}

void Layer::forward()
{
	//_res.print("_res:");
	//_transition.print("_transition:");
	_next->_sum = _transition * _res;
	//_next->_sum.print("_next->_sum:");
	assert(_next->_sum.n_rows == _next->_dimension);
	_next->_validator.apply(_next->_sum, _next->_res);
	//_next->_res.print("_next->_res:");
	assert(_next->_res.n_rows == _next->_dimension);
}

void Layer::backward()
{
	assert(_next);
	assert(_next->_dSum.n_rows == _next->_dimension);
	assert(_transition.n_rows == _next->_dimension);
	assert(_transition.n_cols == _dimension);
	assert(solve(_dSum, _transition, _next->_dSum));
	assert(_dSum.n_rows == _dimension);

	vec reversed(_dimension);
	_validator.reverse(_sum, reversed);
	/*reversed.print("reversed:");
	_transition.print("_transition:");
	_next->_dSum.print("_next->_dSum:");
	_dSum.print("_dSum:");*/
	_validator.eltMult(reversed, _dSum);

	vec ones(_dimension); ones.fill(1.0);
	_dTransition = _next->_dSum * (ones.t() / _res.t());
	/*_res.print("_res:");
	_dTransition.print("_dTransition:");*/
	_transition = _transition + _dTransition;
	//_transition.print("_transition:");
}
