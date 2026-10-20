#include "layer.hpp"
#include "utils.hpp"
#include "settings.hpp"
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ostream>

void FullyConnectedLayer::fill() {
	weights = new __float128*[inputSize];
	for (std::size_t i = 0; i < inputSize; ++i) {
		weights[i] = new __float128[outputSize];
		for (std::size_t j = 0; j < outputSize; ++j) {
			weights[i][j] = randomFloat128()*2-1;
		}
	}

	biases = new __float128[outputSize];
	for (std::size_t j = 0; j < outputSize; ++j) {
		biases[j] = randomFloat128()*2-1;
	}
}

FullyConnectedLayer::FullyConnectedLayer(std::size_t _inputSize, std::size_t _outputSize) {
	inputSize = _inputSize;
	outputSize = _outputSize;

	fill();
}

void FullyConnectedLayer::calc(__float128 *x, __float128 *y) {
	for (std::size_t j = 0; j < outputSize; ++j) {
		y[j] = 0;
		for (std::size_t i = 0; i < inputSize; ++i) {
			y[j] += x[i]*weights[i][j];
		}
	}

	for (std::size_t j = 0; j < outputSize; ++j) {
		y[j] = sigmoid(y[j] + biases[j]);
	}
}

void FullyConnectedLayer::train(__float128 *o, __float128 *z) {
	for (std::size_t i = 0; i < inputSize; ++i) {
		for (std::size_t j = 0; j < outputSize; ++j) {
			weights[i][j] -= SPEED_LEARNING*o[i]*z[j];
		}
	}

	for (std::size_t _j = 0; _j < outputSize; ++_j) {
		biases[_j] -= SPEED_LEARNING*z[_j];
	}
}

void FullyConnectedLayer::sigma(__float128 *x, __float128 *i, __float128 *y) {
	//std::cout << "Sigma" << std::endl;
	for (std::size_t _i = 0; _i < inputSize; ++_i) {
		y[_i] = 0;
		for (std::size_t _j = 0; _j < outputSize; ++_j) {
			y[_i] += i[_i] * (1 - i[_i]) * x[_j] * weights[_i][_j];
		}
	}
}

ConvolutionalLayer::ConvolutionalLayer(std::vector<std::size_t> _inputSize, std::vector<std::size_t> _scanSize, std::size_t _layersCount)
	: inputSizeVector(_inputSize)
	, scanSizeVector(_scanSize)
	, layersCount(_layersCount) {
	inputSize = 1;
	outputSize = layersCount;
	std::size_t scanSize = 1;

	dimensions = _inputSize.size();

	for (std::size_t d = 0; d < dimensions; ++d) {
		inputSize *= inputSizeVector[d];
		outputSize *= inputSizeVector[d] - scanSizeVector[d] + 1;
		scanSize *= scanSizeVector[d];
	}

	weights = new __float128*[scanSize];
	for (std::size_t i = 0; i < scanSize; ++i) {
		weights[i] = new __float128[layersCount];
		for (std::size_t j = 0; j < layersCount; ++j) {
			weights[i][j] = randomFloat128()*2-1;
		}
	}

	biases = new __float128[layersCount];
	for (std::size_t j = 0; j < layersCount; ++j) {
		biases[j] = randomFloat128()*2-1;
	}
}

void ConvolutionalLayer::calc(__float128 *x, __float128 *y) {
	// O(o*d*(i+1)), can be better I think
	std::vector<std::size_t> outputPosition(dimensions+1, 0);
	for (std::size_t j = 0; j < outputSize; ++j) {
		y[j] = 0;
		std::vector<std::size_t> inputPosition(dimensions, 0);
		for (std::size_t i = 0; i < inputSize; ++i) {
			inputPosition[0]++;
			bool in = true;
			std::size_t d;
			for (d = 0; d < dimensions-1; ++d) {
				if (inputPosition[d] < outputPosition[d] || inputPosition[d] >= outputPosition[d] + scanSizeVector[d]) in = false;
			}
			if (inputPosition[d] < outputPosition[d] || inputPosition[d] >= outputPosition[d] + scanSizeVector[d]) in = false;

			if (in) {
				for (d = 0; d < dimensions-1; ++d) {
					y[j] += x[i]*weights[i][j];
				}
			}

			for (std::size_t d = 0; d < dimensions-1; ++d) {
				inputPosition[d+1] += outputPosition[d] / inputSizeVector[d];
				inputPosition[d] %= inputSizeVector[d];
			}
		}
		outputPosition[0]++;
		for (std::size_t d = 0; d < dimensions; ++d) {
			outputPosition[d+1] += outputPosition[d] / (inputSizeVector[d] - scanSizeVector[d] + 1);
			outputPosition[d] %= inputSizeVector[d] - scanSizeVector[d] + 1;
		}
	}

	for (std::size_t j = 0; j < outputSize; ++j) {
		y[j] = sigmoid(y[j] + biases[j]);
	}
}

LayerContainer* createSimpleLayers(std::size_t inputSize, std::size_t outputSize) {
	auto** layers = new Layer*[3] {
		new FullyConnectedLayer(inputSize, inputSize*3),
		new FullyConnectedLayer(inputSize*3, outputSize*3),
		new FullyConnectedLayer(outputSize*3, outputSize),
	};

	return new LayerContainer{layers, 3};
}
