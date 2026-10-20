#pragma once

#include <stdfloat>
#include "layer.hpp"

class NeuralNetwork {
public:
	NeuralNetwork(LayerContainer& _layerContainer);
	void train(__float128* x, __float128* y);
	void calc(__float128* x, __float128* y);
private:
	LayerContainer& layerContainer;
	__float128*** buffer;
};
