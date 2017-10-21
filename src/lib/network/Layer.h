#pragma once

#include <armadillo>


namespace neural {
	namespace details {

		class ValidationFunction;

		class Layer {
		public:
			Layer(size_t dimension_p, ValidationFunction const & validator_p);
			Layer(size_t dimension_p, ValidationFunction const & validator_p, Layer * next_p);
			virtual ~Layer();

			void setWeight(size_t idxInNode_p, size_t idxOutNode_p, float value);
			float getWeight(size_t idxInNode_p, size_t idxOutNode_p);

			float getSum(size_t idxNode_p) const;
			float getRes(size_t idxNode_p) const;
			float getDSum(size_t idxNode_p) const;
			arma::vec getDSum() const;
			void setRes(size_t idxNode_p, float value_p);
			void setDSum(size_t idxNode_p, float value_p);

			void forward();
			void backward();

			arma::mat const & getDTransition() const;
			void updateTransition(arma::mat const & dTransition_p);

		private:
			arma::mat _transition;
			arma::mat _dTransition;
			arma::vec _res;
			arma::vec _sum;
			arma::vec _dSum;

			size_t _dimension;
			ValidationFunction const & _validator;
			Layer * const _next;
		};
	}
}