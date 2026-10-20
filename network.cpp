#include "network.hpp"
#include <cstddef>
#include <iostream>
#include <ostream>

NeuralNetwork::NeuralNetwork(LayerContainer& _layerContainer)
	: layerContainer(_layerContainer) {
	std::cout << "Start data allocation in network" << std::endl;
	std::size_t n = layerContainer.getSize();
	std::size_t i = 0;
	buffer = new __float128**[2];
	buffer[0] = new __float128*[n];
	buffer[1] = new __float128*[n];
	for (auto it = layerContainer.begin(); it != layerContainer.end(); ++it) {
		buffer[0][i] = new __float128[it->getOutputSize()];
		buffer[1][i++] = new __float128[it->getOutputSize()];
	}
	std::cout << "End data allocation in network" << std::endl;
}

void NeuralNetwork::train(__float128 *x, __float128 *y) {
	//std::cout << "Start traning" << std::endl;
	std::size_t n = layerContainer.getSize();
	if (n == 0) return;
	auto it = layerContainer.begin();
	it->calc(x, buffer[0][0]);
	++it;
	for (std::size_t i = 1; it != layerContainer.end(); ++it, ++i) {
		it->calc(buffer[0][i-1], buffer[0][i]);
	}
	for (std::size_t i = 0; i < layerContainer.rbegin()->getOutputSize(); ++i) {
		buffer[1][n-1][i] = (buffer[0][n-1][i] - y[i])*buffer[0][n-1][i]*(1-buffer[0][n-1][i]);
	}
	it = layerContainer.rbegin();
	for (std::size_t i = n-1; it != layerContainer.begin(); --it, --i) {
		it->sigma(buffer[1][i], buffer[0][i-1], buffer[1][i-1]);
		it->train(buffer[0][i-1], buffer[1][i]);
	}
	it->train(x, buffer[1][0]);
	//std::cout << "End training" << std::endl;
}

void NeuralNetwork::calc(__float128 *x, __float128 *y) {
	auto it = layerContainer.begin();
	if (it == layerContainer.end()) return;
	it->calc(x, buffer[0][0]);
	++it;
	std::size_t i = 1;
	for (; it != layerContainer.rbegin(); ++it) {
		it->calc(buffer[0][i-1], buffer[0][i]);
		++i;
	}
	it->calc(buffer[0][i-1], y);
}
