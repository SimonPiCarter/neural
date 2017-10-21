#pragma once

#include <vector>

#include "ValidationFunction.h"

namespace neural {
	namespace details{
		class Layer;
	}

	class Network {
	public:
		Network(size_t depth_p, size_t width_p, size_t inputSize_p, size_t outputSize_p);
		~Network();

		void forward(std::vector<float> const & input_p);
		void backward(std::vector<float> const & expectedOutput_p);

		std::vector<float> getOutput() const;
		std::vector<float> getInput() const;
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

		details::ValidationFunction _validator;
	};
}