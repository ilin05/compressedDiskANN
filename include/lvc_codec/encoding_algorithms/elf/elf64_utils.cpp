#include "elf64_utils.h"

namespace encoding_algorithm {
namespace elf {

const int Elf64Utils::F_ALPHA[21] = {
	0, 4, 7, 10, 14, 17, 20, 24, 27, 30,
	34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67
};

const double Elf64Utils::MAP_10_P[21] = {
	1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6,
	1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12,
	1.0e13, 1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18,
	1.0e19, 1.0e20
};

const double Elf64Utils::MAP_10_N[21] = {
	1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
	1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12,
	1.0e-13, 1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17,
	1.0e-18, 1.0e-19, 1.0e-20
};

const long long Elf64Utils::MAP_SP_GREATER_1[10] = {
	1LL, 10LL, 100LL, 1000LL, 10000LL,
	100000LL, 1000000LL, 10000000LL, 100000000LL, 1000000000LL
};

const double Elf64Utils::MAP_SP_LESS_1[11] = {
	1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
	1e-6, 1e-7, 1e-8, 1e-9, 1e-10
};

const double Elf64Utils::LOG_2_10 = 3.3219280948873626;

}
}
