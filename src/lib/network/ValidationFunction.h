#pragma once

#include <armadillo>

namespace neural {
	namespace details {
		class ValidationFunction
		{
		public:
			void eltMult(arma::vec const & mult_p, arma::vec & out_p) const;
			void apply(arma::vec const & sum_p, arma::vec & res_p) const;
			void reverse(arma::vec const & sum_p, arma::vec & res_p) const;

			virtual float apply(float const & x) const = 0;
			virtual float reverse(float const & x) const = 0;
		};

		class SigmoidValidationFunction : public ValidationFunction
		{
		public:
			virtual float apply(float const & x) const;
			virtual float reverse(float const & x) const;
		};

		class IdentityValidationFunction : public ValidationFunction
		{
		public:
			virtual float apply(float const & x) const;
			virtual float reverse(float const & x) const;
		};
	}
}
