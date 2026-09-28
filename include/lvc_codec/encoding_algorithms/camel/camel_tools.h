#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>

namespace encoding_algorithm {
namespace camel {

    struct CamelTools {
        static const double EPS;
        static const int COST[16];
        static const double I_POW2[5];
        static const long long POW10[5];

        static double quick_pow2(int exp) {
            if (exp >= 0) {
                return std::pow(2.0, static_cast<double>(exp));
            }
            int idx = -exp;
            if (idx >= 0 && idx < 5) {
                return I_POW2[idx];
            }
            return std::pow(2.0, static_cast<double>(exp));
        }

        static long long quick_pow10(int exp) {
            if (exp >= 0 && exp < 5) {
                return POW10[exp];
            }
            return static_cast<long long>(std::llround(std::pow(10.0, static_cast<double>(exp))));
        }

        static int acc_decimal_count(double value, int lim) {
            double factor = 1.0;
            for (int i = 0; i < lim; ++i) {
                double scaled = value * factor;
                double rounded = std::round(scaled);
                if (std::fabs(scaled - rounded) < EPS) {
                    return i;
                }
                factor *= 10.0;
            }
            return lim;
        }

        static int decimal_count(double value) {
            int decimals = acc_decimal_count(value, 5);
            decimals = std::max(1, decimals);
            return std::min(4, decimals);
        }

        static double calculate_dxor(double dec, int l) {
            double pow = quick_pow2(-l);
            if (pow == 0.0) {
                return dec;
            }
            return dec - pow * std::floor(dec / pow);
        }

        static int calculate_max(int l) {
            if (l < 0) {
                return 0;
            }
            if (l >= 16) {
                return COST[15];
            }
            return COST[l];
        }

        static long long truncate(double value) {
            if (value > EPS) return static_cast<long long>(std::floor(value));
            if (value < -EPS) return static_cast<long long>(std::ceil(value));
            return 0;
        }
    };

}
}
