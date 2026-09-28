#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <cstdint>

namespace encoding_algorithm {
namespace dexor {

    namespace DeXORTools {
        const double EQUAL_EPS = 1e-23;
        const double INTEGER_EPS = 1e-6;
        const int OFF = 23;

        const std::vector<int> COST = {0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67, 70, 74, 77};
        const std::vector<double> P10 = {1e-23, 1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17,
            1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
            1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
            1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23};
        const std::vector<long long> P10L = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000L, 100000000000L, 1000000000000L, 10000000000000L, 100000000000000L, 1000000000000000L, 10000000000000000L, 100000000000000000L};
        const std::vector<int> P2 = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

        inline double getP10(int pow) {
            return P10[pow + OFF];
        }

        inline long long getP10L(int pow) {
            return P10L[pow];
        }

        inline int getP2(int pow) {
            return P2[pow];
        }

        inline int comp(double a, double b, double eps) {
            double delta = a - b;
            if (delta >= eps) return 1;
            if (delta <= -eps) return -1;
            return 0;
        }

        inline int comp(double a, double b) {
            return comp(a, b, EQUAL_EPS);
        }

        inline bool isInt(double value, double eps) {
            return comp(value, std::round(value), eps) == 0;
        }
        
        inline bool isInt(double value) {
            return isInt(value, EQUAL_EPS);
        }

        inline bool isEnd(double value, int end) {
            double alpha = value / getP10(end);
            double beta = value / getP10(end - 1);
            return isInt(alpha) && !isInt(beta);
        }

        inline int getEnd_HP(double value, int last_end) {
            if (isEnd(value, last_end)) return last_end;
            std::string s = std::to_string(value);
            size_t index = s.find('.');
            if (index == std::string::npos) {
                int e = 0;
                for (int i = s.length() - 1; i >= 0; i--) {
                    if (s[i] == '0') e++;
                    else break;
                }
                return e;
            } else {
                return index - (s.length() - 1);
            }
        }

        inline int getEnd(double value, int last_end) {
            if (comp(value, 0, EQUAL_EPS) == 0) return 0;
            if (last_end < -12) return getEnd_HP(value, last_end);
            int q = last_end;
            double vq = value / getP10(q);
            if (isInt(vq, INTEGER_EPS)) {
                vq = value / getP10(q + 1);
                while (isInt(vq, INTEGER_EPS)) {
                    q++;
                    vq = value / getP10(q + 1);
                }
                return q;
            } else {
                q--;
                vq = value / getP10(q);
                while (!isInt(vq, INTEGER_EPS)) {
                    q--;
                    vq = value / getP10(q);
                }
                return q;
            }
        }

        inline int decimalBits(int dp) {
            return COST[dp];
        }

        inline long long truncate(double value) {
            if (isInt(value)) return std::round(value);
            if (value > EQUAL_EPS) return static_cast<long long>(std::floor(value + INTEGER_EPS));
            if (value < -EQUAL_EPS) return static_cast<long long>(std::ceil(value - INTEGER_EPS));
            return 0;
        }

        inline long long segment(long long v, int st, int ed) { // 1-based index
            int len = ed - st + 1;
            long long mask = (1LL << len) - 1;
            return (v >> (64 - ed)) & mask;
        }

        inline double epsilon(double value) {
            union { double d; long long l; } u;
            u.d = value;
            long long exp = segment(u.l, 2, 12);
            u.l = (exp - 52) << 52;
            return u.d;
        }
    }
}
}
