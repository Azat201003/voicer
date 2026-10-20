#include "network.hpp"
#include <array>
#include <iostream>
#include <vector>

using V5 = std::array<__float128, 5>;
using V1 = std::array<__float128, 1>;

int main() {
	auto* net = new NeuralNetwork(*createSimpleLayers(5, 1));
	// a and b or not c
	std::vector<std::pair<V5, V1>> data {
		{{1, 1, 1, 2, 3}, {1}},
		{{20, 1, 1, 2, 3}, {0}},
		{{1, 67, 1, 2, 3}, {0.25}},
		{{1, 67, 128, 2, 3}, {0.5}},
		{{9, 1, 2, 3, 4}, {0}},
		{{1, 9, 2, 3, 4}, {0.25}},
		{{1, 2, 9, 3, 4}, {0.5}},
		{{1, 2, 3, 9, 4}, {0.75}},
		{{1, 2, 3, 4, 9}, {1}},
		{{-1, -2, -3, -4, -5}, {0}},
		{{-5, -1, -2, -3, -4}, {0.25}},
		{{-5, -4, -1, -2, -3}, {0.5}},
		{{-5, -4, -3, -1, -2}, {0.75}},
		{{-5, -4, -3, -2, -1}, {1}},
		{{100, -50, 25, 0, -75}, {0}},
		{{-100, 50, 25, 0, -75}, {0.25}},
		{{-100, -50, 25, 0, -75}, {0.5}},
		{{-100, -50, -25, 0, -75}, {0.75}},
		{{-100, -50, -25, -10, 0}, {1}},
		{{2147483647, 0, -1, 2, 3}, {0}},
		{{5, 8, 7, 6, 4}, {0.25}},
		{{5, 6, 8, 7, 4}, {0.5}},
		{{5, 6, 7, 8, 4}, {0.75}},
		{{5, 6, 7, 8, 10}, {1}},
		{{1, -67, -128, 2, -3}, {0.75}},
	};
	for (int epoch = 0; epoch < 1024; ++epoch) {
		for (auto [x, y] : data) {
			net->train(x.data(), y.data());
		}
	}

	while (true) {
		float a, b, c, d, e;
		std::cin >> a >> b >> c >> d >> e;
		__float128* inp = new __float128[5];
		inp[0] = a; inp[1] = b; inp[2] = c; inp[3] = d; inp[4] = e;
		__float128* out = new __float128();
		net->calc(inp, out);
		std::cout << (float)*out << std::endl;
	}
}
