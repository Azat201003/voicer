#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <vector>

class Layer {
public:
	// y - output
	virtual void calc(__float128* x, __float128* y) = 0;
	// o - output, z - sigma
	virtual void train(__float128* o, __float128* z) = 0;
	// x - previous sigma, y - current
	virtual void sigma(__float128* x, __float128* i, __float128* y) = 0;
	std::size_t getInputSize() { return inputSize; }
	std::size_t getOutputSize() { return outputSize; }
protected:
	std::size_t inputSize;
	std::size_t outputSize;
};

class ConvolutionalLayer : public Layer {
public:
	ConvolutionalLayer(std::vector<std::size_t> _inputSize, std::vector<std::size_t> _scanSize, std::size_t _layersCount);

	void calc(__float128* x, __float128* y) override;
	void train(__float128* o, __float128* z) override;
	void sigma(__float128* x, __float128* i, __float128* y) override;
private:
	std::vector<std::size_t> inputSizeVector, scanSizeVector;
	std::size_t dimensions;
	std::size_t layersCount;
	__float128** weights;
	__float128* biases;
	__float128** weightsChange;
	__float128* biasesChange;
};

class PoolingLayer : public Layer {
public:
	enum class Type {
		MAX,
		AVERAGE,
	};

	PoolingLayer(std::vector<std::size_t> _inputSize, std::vector<std::size_t> _scanSize, Type _type);

	void calc(__float128* x, __float128* y) override;
	void train(__float128* o, __float128* z) override;
	void sigma(__float128* x, __float128* i, __float128* y) override;
private:
	std::vector<std::size_t> inputSizeVector, scanSizeVector;
	Type type;
};

class FullyConnectedLayer : public Layer {
public:
	FullyConnectedLayer(std::size_t _inputSize, std::size_t _outputSize);

	void calc(__float128* x, __float128* y) override;
	void train(__float128* o, __float128* z) override;
	void sigma(__float128* x, __float128* i, __float128* y) override;
	void fill();
private:
	__float128** weights;
	__float128* biases;
	__float128** weightsChange;
	__float128* biasesChange;
};

class LayerContainerBuilder;

struct LayerContainer {
	struct iterator {
		iterator& operator++() {
			++i;
			return *this;
		}
		iterator& operator--() {
			--i;
			return *this;
		}
		bool operator==(const iterator& other) {
			return other.i == i;
		}
		bool operator!=(const iterator& other) {
			return other.i != i;
		}
		Layer& operator*() {
			return **i;
		}
		Layer* operator->() {
			return *i;
		}
	private:
		Layer** i;
		iterator(Layer** _i) : i(_i) {}
		friend struct LayerContainer;
	};

	explicit LayerContainer(Layer** _layers, std::size_t _size) : layers(_layers), size(_size) {}
	[[nodiscard]] iterator begin() {
		return iterator{layers};
	}
	[[nodiscard]] iterator rbegin() {
		return iterator{layers+size-1};
	}
	[[nodiscard]] iterator end() {
		return iterator{layers+size};
	}
	[[nodiscard]] iterator rend() {
		return iterator{layers-1};
	}
	std::size_t getSize() {
		return size;
	}
private:
	Layer** layers;
	std::size_t size;
};

LayerContainer* createSimpleLayers(std::size_t inputSize, std::size_t outputSize);
