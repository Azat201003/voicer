#include "network.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <vector>

using V3 = std::array<__float128, 3>;
using V1 = std::array<__float128, 1>;

int main() {
	auto* net = new NeuralNetwork(*createSimpleLayers(3, 1));
	// a and b or not c
	std::vector<std::pair<V3, V1>> data {
		{{0, 0, 0}, {1}},
		{{0, 0, 1}, {0}},
		{{0, 1, 0}, {1}},
		{{0, 1, 1}, {0}},
		{{1, 0, 0}, {1}},
		{{1, 0, 1}, {0}},
		{{1, 1, 0}, {1}},
	};

	auto rng = std::default_random_engine{};
	for (int epoch = 0; epoch < 64*1024; ++epoch) {
		std::shuffle(data.begin(), data.end(), rng);
		for (auto [x, y] : data) {
			net->train(x.data(), y.data());
		}
	}

	while (true) {
		float a, b, c;
		std::cin >> a >> b >> c;
		__float128* inp = new __float128[3];
		inp[0] = a; inp[1] = b; inp[2] = c;
		__float128* out = new __float128();
		net->calc(inp, out);
		std::cout << (float)*out << std::endl;
	}
}
