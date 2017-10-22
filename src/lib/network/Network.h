#pragma once

#include <vector>

#include "ValidationFunction.h"

namespace neural {
	namespace details{
		class Layer;

		struct TrainingData {
			std::vector<double> input;
			std::vector<double> expectedOutput;
		};
	}

	class Network {
	public:
		Network(size_t depth_p, size_t width_p, size_t inputSize_p, size_t outputSize_p);
		~Network();

		void forward(std::vector<double> const & input_p);
		void backward(std::vector<double> const & expectedOutput_p);

		void train(std::vector<details::TrainingData> const & trainingData_p, size_t iterMax_p=20000);

		std::vector<double> getOutput() const;
		std::vector<double> getInput() const;
		/**
		 * Layer 0 is input, Layer _depth-1 is output
		 */
		details::Layer * getLayer(size_t depth_p);

	private:
		size_t _depth;
		size_t _width;
		size_t _inputSize;
		size_t _outputSize;
		std::vector<details::Layer *> _layers;

		details::SigmoidValidationFunction _validator;
		details::IdentityValidationFunction _identity;
	};
}