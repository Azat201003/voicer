#pragma once

#include <random>
#include <cstdint>
#include <quadmath.h>

std::mt19937_64 rng{67};

// 0 <= ans <= 1
__float128 randomFloat128() {
	static const __float128 S64  = (__float128)0x1p-64;
    static const __float128 S128 = (__float128)0x1p-128;
    uint64_t hi = rng();
    uint64_t lo = rng();
    return (__float128)hi * S64 + (__float128)lo * S128;
}

__float128 sigmoid(__float128 x)
{
    if (x >= 0) {
        __float128 e = expq(-x);
        return (__float128)1 / ((__float128)1 + e);
    } else {
        __float128 e = expq(x);
        return e / ((__float128)1 + e);
    }
}
