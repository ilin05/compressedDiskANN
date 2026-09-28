#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace utils {
namespace binary_tools {

    inline int getLen(int x) {
        if (x == 0) return 1;
        if (x < 0) x = -x;
        return floor(log2(x)) + 1;
    }

    inline int getLen(long long x) {
        if (x == 0) return 1;
        if (x < 0) x = -x;
        return floor(log2(x)) + 1;
    }

    inline long long xor_double(double a, double b) {
        union { double d; long long l; } u_a, u_b;
        u_a.d = a;
        u_b.d = b;
        return u_a.l ^ u_b.l;
    }

    inline double xor_double(long long a, double b) {
        union { double d; long long l; } u_b;
        u_b.d = b;
        u_b.l ^= a;
        return u_b.d;
    }

    inline int leadZeros(long long val, int size) {
        if (val == 0) return size;
        // Use built-in functions for efficiency
#if defined(__GNUC__) || defined(__clang__)
        int total_leading_zeros = __builtin_clzll(val);
#elif defined(_MSC_VER)
        unsigned long index;
        _BitScanReverse64(&index, val);
        int total_leading_zeros = 63 - index;
#else
        // Fallback for other compilers
        int total_leading_zeros = 0;
        for (int i = 63; i >= 0; --i) {
            if (((val >> i) & 1) == 0) {
                total_leading_zeros++;
            } else {
                break;
            }
        }
#endif
        return std::max(0, total_leading_zeros - (64 - size));
    }

    inline int tailZeros(long long val, int size) {
        if (val == 0) return size;
#if defined(__GNUC__) || defined(__clang__)
        return std::min(size, __builtin_ctzll(val));
#elif defined(_MSC_VER)
        unsigned long index;
        _BitScanForward64(&index, val);
        return std::min(size, (int)index);
#else
        // Fallback for other compilers
        int count = 0;
        for (int i = 0; i < size; ++i) {
            if (((val >> i) & 1) == 0) {
                count++;
            } else {
                break;
            }
        }
        return count;
#endif
    }

    inline int CBL(long long v, int size){
        if (v == 0) return 0;
        int lead = leadZeros(v, size);
        int tail = tailZeros(v, size);
        return size - lead - tail;
    }

} // namespace binary_tools
} // namespace utils
