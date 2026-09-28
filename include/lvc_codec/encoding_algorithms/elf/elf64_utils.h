#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace encoding_algorithm {
namespace elf {

    class Elf64Utils {
    private:
        static const int F_ALPHA[21];
        static const double MAP_10_P[21];
        static const double MAP_10_N[21];
        static const long long MAP_SP_GREATER_1[10];
        static const double MAP_SP_LESS_1[11];
        static const double LOG_2_10;

        static std::pair<int, int> getSPAnd10iNFlag(double v) {
            std::pair<int, int> result{0, 0};
            if (v >= 1.0) {
                for (std::size_t i = 0; i < 9; ++i) {
                    if (v < static_cast<double>(MAP_SP_GREATER_1[i + 1])) {
                        result.first = static_cast<int>(i);
                        return result;
                    }
                }
            } else {
                for (std::size_t i = 1; i < 11; ++i) {
                    if (v >= MAP_SP_LESS_1[i]) {
                        result.first = -static_cast<int>(i);
                        result.second = (v == MAP_SP_LESS_1[i]) ? 1 : 0;
                        return result;
                    }
                }
            }
            double log10v = std::log10(v);
            result.first = static_cast<int>(std::floor(log10v));
            result.second = (log10v == std::floor(log10v)) ? 1 : 0;
            return result;
        }

        static int getSignificantCount(double v, int sp, int lastBetaStar) {
            int i;
            if (lastBetaStar != std::numeric_limits<int>::max() && lastBetaStar != 0) {
                i = std::max(lastBetaStar - sp - 1, 1);
            } else if (lastBetaStar == std::numeric_limits<int>::max()) {
                i = 17 - sp - 1;
            } else if (sp >= 0) {
                i = 1;
            } else {
                i = -sp;
            }

            double temp = v * get10iP(i);
            double temp_floor = std::floor(temp);
            while (temp_floor != temp) {
                ++i;
                temp = v * get10iP(i);
                temp_floor = std::floor(temp);
            }

            if (temp / get10iP(i) != v) {
                return 17;
            }

            while (i > 0 && std::fmod(temp_floor, 10.0) == 0.0) {
                --i;
                temp_floor /= 10.0;
            }
            return sp + i + 1;
        }

        static double get10iP(int i) {
            if (i < 0) {
                throw std::invalid_argument("The argument should be greater than 0");
            }
            if (i >= 21) {
                return std::pow(10.0, static_cast<double>(i));
            }
            return MAP_10_P[static_cast<std::size_t>(i)];
        }

    public:
        static int getFAlpha(int alpha) {
            if (alpha < 0) {
                throw std::invalid_argument("The argument should be greater than 0");
            }
            if (alpha >= 21) {
                return static_cast<int>(std::ceil(alpha * LOG_2_10));
            }
            return F_ALPHA[static_cast<std::size_t>(alpha)];
        }

        static std::pair<int, int> getAlphaAndBetaStar(double v, int lastBetaStar) {
            if (v < 0) {
                v = -v;
            }
            auto spAndFlag = getSPAnd10iNFlag(v);
            int beta = getSignificantCount(v, spAndFlag.first, lastBetaStar);
            int alpha = beta - spAndFlag.first - 1;
            int betaStar = (spAndFlag.second == 1) ? 0 : beta;
            return {alpha, betaStar};
        }

        static double roundUp(double v, int alpha) {
            double scale = get10iP(alpha);
            if (v < 0) {
                return std::floor(v * scale) / scale;
            }
            return std::ceil(v * scale) / scale;
        }

        static double get10iN(int i) {
            if (i < 0) {
                throw std::invalid_argument("The argument should be greater than 0");
            }
            if (i >= 21) {
                return std::pow(10.0, -static_cast<double>(i));
            }
            return MAP_10_N[static_cast<std::size_t>(i)];
        }

        static int getSP(double v) {
            if (v >= 1.0) {
                for (std::size_t i = 0; i < 9; ++i) {
                    if (v < static_cast<double>(MAP_SP_GREATER_1[i + 1])) {
                        return static_cast<int>(i);
                    }
                }
            } else {
                for (std::size_t i = 1; i < 11; ++i) {
                    if (v >= MAP_SP_LESS_1[i]) {
                        return -static_cast<int>(i);
                    }
                }
            }
            return static_cast<int>(std::floor(std::log10(v)));
        }
    };

}
}
