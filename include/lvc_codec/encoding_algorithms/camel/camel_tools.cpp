#include "camel_tools.h"

namespace encoding_algorithm {
namespace camel {

const double CamelTools::EPS = 1e-5;
const int CamelTools::COST[16] = {
	0, 4, 7, 10, 14, 17, 20, 24,
	27, 30, 34, 37, 40, 44, 47, 50
};
const double CamelTools::I_POW2[5] = {1.0, 0.5, 0.25, 0.125, 0.0625};
const long long CamelTools::POW10[5] = {1LL, 10LL, 100LL, 1000LL, 10000LL};

}
}
