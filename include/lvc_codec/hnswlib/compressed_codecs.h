#pragma once

#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../examples/utils/memory_stream_writer.h"
#include "../examples/utils/memory_block_stream_reader.h"
#include "../examples/utils/binary_tools.h"
#include "../encoding_algorithms/dexor/dexor_tools.h"
#include "../encoding_algorithms/camel/camel_tools.h"
#include "../encoding_algorithms/elf/elf64_utils.h"

namespace hnswlib {
namespace codecs {

enum class AlgorithmType {
    DeXOR,
    Gorilla,
    Elf,
    Camel,
    DeXORPlus
};

inline AlgorithmType getAlgorithmType(const std::string& name) {
    if (name == "Gorilla") return AlgorithmType::Gorilla;
    if (name == "Elf") return AlgorithmType::Elf;
    if (name == "Camel") return AlgorithmType::Camel;
    if (name == "DeXORPlus") return AlgorithmType::DeXORPlus;
    return AlgorithmType::DeXOR; // default
}

struct DeXORState {
    double current_value;
    struct {
        double previous_value;
        int previous_q;
        int previous_delta;
        long long previous_exp;
        int EL;
        int contract_step;
        int rho;
    } dexor;
    DeXORState() { reset(); }
    void reset() {
        current_value = 0.0;
        dexor.previous_value = 0.0;
        dexor.previous_q = 0;
        dexor.previous_delta = 0;
        dexor.previous_exp = 1023;
        dexor.EL = 1;
        dexor.contract_step = 0;
        dexor.rho = 1;
    }
};

struct GorillaState {
    double current_value;
    struct {
        double previous_value;
        int previous_lead;
        int previous_tail;
        bool first;
    } gorilla;
    GorillaState() { reset(); }
    void reset() {
        current_value = 0.0;
        gorilla.previous_value = 0.0;
        gorilla.previous_lead = 0;
        gorilla.previous_tail = 0;
        gorilla.first = true;
    }
};

struct ElfState {
    double current_value;
    struct {
        long long previous_long_value;
        int previous_lead;
        int previous_tail;
        int previous_betaStar;
        bool first;
    } elf;
    ElfState() { reset(); }
    void reset() {
        current_value = 0.0;
        elf.previous_long_value = 0;
        elf.previous_lead = 0;
        elf.previous_tail = 0;
        elf.previous_betaStar = 0;
        elf.first = true;
    }
};

struct CamelState {
    double current_value;
    struct {
        long long previous_integer;
        bool first;
    } camel;
    CamelState() { reset(); }
    void reset() {
        current_value = 0.0;
        camel.previous_integer = 0;
        camel.first = true;
    }
};

struct DeXORPlusState {
    double current_value;
    struct {
        double previous_value;
        int p_q;
        int p_o;
        int p_delta;
        int epsilon;
    } dexorplus;
    
    static int& default_epsilon() { static int eps = -4; return eps; }
    
    DeXORPlusState() { reset(default_epsilon()); }
    DeXORPlusState(int eps) { reset(eps); }
    void reset(int eps = default_epsilon()) {
        current_value = 0.0;
        dexorplus.previous_value = 0.0;
        dexorplus.p_q = 0;
        dexorplus.p_o = 0;
        dexorplus.p_delta = 0;
        dexorplus.epsilon = eps;
    }
};

class CompressedCodec {
public:
    //     return 0.0;
    // }

public:
    // --------- DeXOR ---------
    template <typename StateT>
    static inline void encode_dexor(double value, StateT& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::dexor;
        int q = DeXORTools::getEnd(value, state.dexor.previous_q);
        int delta = 0;
        double alpha = 0;
        while (delta < 16) {
            double pow = DeXORTools::getP10(q + delta);
            long long a = DeXORTools::truncate(value / pow);
            long long b = DeXORTools::truncate(state.dexor.previous_value / pow);
            if (a == b) {
                alpha = a * pow;
                break;
            }
            delta++;
        }

        double pow = DeXORTools::getP10(q);
        double residual = value - alpha;
        long long beta = std::llround(residual / pow);

        if (delta >= 16 || DeXORTools::comp(alpha + beta * pow, value, pow) != 0) {
            writer.write(true);
            writer.write(true);
            dexor_exception_handle(value, state, writer);
            return;
        }

        beta = std::llabs(beta);
        bool flag = q == state.dexor.previous_q;
        if (flag && delta == state.dexor.previous_delta) {
            writer.write(true);
            writer.write(false);
        } else {
            writer.write(false);
            writer.write(flag);
            if (!flag) {
                writer.write(q + 20, 5);
                state.dexor.previous_q = q;
            }
            writer.write(delta, 4);
            state.dexor.previous_delta = delta;
        }

        if (DeXORTools::comp(alpha, 0) == 0) {
            writer.write(value > 0);
        }

        writer.write(beta, DeXORTools::decimalBits(delta));
        state.dexor.previous_value = value;
    }

    template <typename StateT>
    static inline void dexor_exception_handle(double value, StateT& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::dexor;
        union { double d; long long l; } u;
        u.d = value;
        long long exp = DeXORTools::segment(u.l, 2, 12);
        long long delta = exp - state.dexor.previous_exp;
        int bias = DeXORTools::getP2(state.dexor.EL - 1) - 1;

        if (delta >= -bias && delta <= bias) {
            writer.write(delta + bias, state.dexor.EL);
            writer.write(u.l < 0);
            writer.write(u.l, 52);

            if (state.dexor.EL > 1) {
                int su_bias = DeXORTools::getP2(state.dexor.EL - 2) - 1;
                if(delta >= -su_bias && delta <= su_bias) {
                    state.dexor.contract_step++;
                } else {
                    state.dexor.contract_step = 0;
                }
                if (state.dexor.contract_step == state.dexor.rho) {
                    state.dexor.EL--;
                    state.dexor.contract_step = 0;
                }
            }
        } else {
            writer.write(DeXORTools::getP2(state.dexor.EL) - 1, state.dexor.EL);
            writer.write(u.l, 64);
            state.dexor.contract_step = 0;
            if (state.dexor.EL < 10) state.dexor.EL++;
        }
        state.dexor.previous_exp = exp;
    }

    template <typename StateT>
    static inline double decode_dexor(StateT& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::dexor;
        int con = reader.readInt(2);
        if (con == 3) {
            return dexor_exception_decode(state, reader);
        }
        if (con == 0 || con == 1) {
            if (con == 0) state.dexor.previous_q = reader.readInt(5) - 20;
            state.dexor.previous_delta = reader.readInt(4);
        }
        
        const double pow_val = DeXORTools::getP10(state.dexor.previous_q + state.dexor.previous_delta);
        const double truncated = static_cast<double>(DeXORTools::truncate(state.dexor.previous_value / pow_val));
        double previous_alpha = truncated * pow_val;

        long long sign = previous_alpha > 0 ? 1LL : -1LL;
        if (DeXORTools::comp(previous_alpha, 0) == 0) {
            sign = reader.readBoolean() ? 1LL : -1LL;
        }

        const int bits = DeXORTools::decimalBits(state.dexor.previous_delta);
        long long magnitude = bits > 0 ? reader.readLong(bits) : 0;
        const double beta = static_cast<double>(sign * magnitude) * DeXORTools::getP10(state.dexor.previous_q);

        state.dexor.previous_value = previous_alpha + beta;
        return state.dexor.previous_value;
    }

    template <typename StateT>
    static inline double dexor_exception_decode(StateT& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::dexor;
        const int bias = DeXORTools::getP2(state.dexor.EL - 1) - 1;
        const long long delta = reader.readLong(state.dexor.EL) - bias;
        uint64_t lv = 0;

        if (delta >= -bias && delta <= bias) {
            state.dexor.previous_exp += delta;
            const uint64_t sign = static_cast<uint64_t>(reader.readLong(1));
            const uint64_t mantissa = static_cast<uint64_t>(reader.readLong(52));
            lv = (sign << 63) | (static_cast<uint64_t>(state.dexor.previous_exp) << 52) | mantissa;

            if (state.dexor.EL > 1) {
                const int su_bias = DeXORTools::getP2(state.dexor.EL - 2) - 1;
                if (delta >= -su_bias && delta <= su_bias) {
                    state.dexor.contract_step++;
                } else {
                    state.dexor.contract_step = 0;
                }
                if (state.dexor.contract_step == state.dexor.rho) {
                    state.dexor.EL--;
                    state.dexor.contract_step = 0;
                }
            }
        } else {
            lv = static_cast<uint64_t>(reader.readLong(64));
            state.dexor.previous_exp = DeXORTools::segment(static_cast<long long>(lv), 2, 12);
            if (state.dexor.EL < 10) {
                state.dexor.EL++;
                state.dexor.contract_step = 0;
            }
        }

        union { uint64_t bits; double value; } converter{};
        converter.bits = lv;
        return converter.value;
    }

    // --------- Gorilla ---------
    template <typename StateT>
    static inline void encode_gorilla(double value, StateT& state, utils::MemoryStreamWriter& writer) {
        if (state.gorilla.first) {
            writer.write(value, 64);
            state.gorilla.first = false;
        } else {
            if (value == state.gorilla.previous_value) {
                writer.write(true);
            } else {
                writer.write(false);
                long long xor_val = utils::binary_tools::xor_double(value, state.gorilla.previous_value);
                int lead = utils::binary_tools::leadZeros(xor_val, 64);
                int tail = utils::binary_tools::tailZeros(xor_val, 64);
                if (lead >= state.gorilla.previous_lead && tail >= state.gorilla.previous_tail) {
                    writer.write(true);
                    int len = 64 - state.gorilla.previous_lead - state.gorilla.previous_tail;
                    writer.write(static_cast<long long>(static_cast<unsigned long long>(xor_val) >> state.gorilla.previous_tail), len);
                } else {
                    writer.write(false);
                    int lim_lead = std::min(lead, 31);
                    writer.write(lim_lead, 5);
                    int len = 64 - lim_lead - tail;
                    writer.write(len - 1, 6);
                    writer.write(static_cast<long long>(static_cast<unsigned long long>(xor_val) >> tail), len);
                }
                state.gorilla.previous_lead = lead;
                state.gorilla.previous_tail = tail;
            }
        }
        state.gorilla.previous_value = value;
    }

    template <typename StateT>
    static inline double decode_gorilla(StateT& state, utils::MemoryBlockStreamReader& reader) {
        if (state.gorilla.first) {
            state.gorilla.previous_value = reader.readDouble(64);
            state.gorilla.first = false;
        } else {
            if (reader.readBoolean()) {
                return state.gorilla.previous_value;
            }
            long long xor_val = 0;
            if (reader.readBoolean()) {
                int len = 64 - state.gorilla.previous_lead - state.gorilla.previous_tail;
                xor_val = reader.readLong(len) << state.gorilla.previous_tail;
            } else {
                int lim_lead = reader.readInt(5);
                int len = reader.readInt(6) + 1;
                int tail = 64 - len - lim_lead;
                xor_val = reader.readLong(len) << tail;
            }
            state.gorilla.previous_lead = utils::binary_tools::leadZeros(xor_val, 64);
            state.gorilla.previous_tail = utils::binary_tools::tailZeros(xor_val, 64);
            state.gorilla.previous_value = utils::binary_tools::xor_double(xor_val, state.gorilla.previous_value);
        }
        return state.gorilla.previous_value;
    }

    // --------- Elf ---------
    template <typename StateT>
    static inline void encode_elf(double value, StateT& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::elf;
        if (state.elf.first) {
            writer.write(value, 64);
            union { double d; unsigned long long ull; } u; u.d = value;
            state.elf.previous_long_value = u.ull;
            state.elf.first = false;
            return;
        }

        union { double d; unsigned long long ull; } u; u.d = value;
        if (value == 0.0 || std::isinf(value)) {
            writer.write(false);
        } else {
            auto alphaBeta = Elf64Utils::getAlphaAndBetaStar(value, state.elf.previous_betaStar); 
            int exponent = static_cast<int>((u.ull >> 52) & 0x7FFULL);      
            int gAlpha = Elf64Utils::getFAlpha(alphaBeta.first) + exponent - 1023;  
            int eraseBits = 52 - gAlpha;
            int betaStar = alphaBeta.second;

            unsigned long long mask;
            if (eraseBits <= 0) mask = std::numeric_limits<unsigned long long>::max();
            else mask = std::numeric_limits<unsigned long long>::max() << (static_cast<unsigned int>(eraseBits) & 0x3F);     

            unsigned long long delta = (~mask) & u.ull;
            if (betaStar < 16 && delta != 0 && eraseBits > 4) {
                writer.write((betaStar | 0x10), 5);
                state.elf.previous_betaStar = betaStar;
                u.ull = mask & u.ull;
            } else {
                writer.write(false);
            }
        }
        long long current = u.ull;
        unsigned long long xor_val = u.ull ^ static_cast<unsigned long long>(state.elf.previous_long_value);

        int lead = std::min(utils::binary_tools::leadZeros(static_cast<long long>(xor_val), 64), 7);
        int tail = utils::binary_tools::tailZeros(static_cast<long long>(xor_val), 64);

        if (xor_val != 0 && lead == state.elf.previous_lead && tail >= state.elf.previous_tail) {
            writer.write(0, 2);
            int len = 64 - state.elf.previous_lead - state.elf.previous_tail;
            writer.write(static_cast<long long>(xor_val >> state.elf.previous_tail), len);
        } else if (xor_val == 0) {
            writer.write(1, 2);
        } else {
            writer.write(true);
            int len = 64 - lead - tail;
            if (len <= 16) {
                writer.write(false); writer.write(lead, 3); writer.write(len - 1, 4);
            } else {
                writer.write(true); writer.write(lead, 3); writer.write(len - 1, 6);
            }
            writer.write(static_cast<long long>(xor_val >> tail), len);
        }

        state.elf.previous_lead = lead;
        state.elf.previous_tail = tail;
        state.elf.previous_long_value = current;
    }

    template <typename StateT>
    static inline double decode_elf(StateT& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::elf;
        if (state.elf.first) {
            double v = reader.readDouble(64);
            union { double d; long long ll; } u; u.d = v;
            state.elf.previous_long_value = u.ll;
            state.elf.first = false;
            return v;
        }

        bool c1 = reader.readBoolean();
        int betaStar = 0;
        if (c1) betaStar = reader.readInt(4);

        int c2 = reader.readInt(2);
        long long xor_val = 0;
        if (c2 == 0) {
            int len = 64 - state.elf.previous_lead - state.elf.previous_tail;
            xor_val = reader.readLong(len) << state.elf.previous_tail;
            state.elf.previous_lead = std::min(utils::binary_tools::leadZeros(xor_val, 64), 7);
            state.elf.previous_tail = utils::binary_tools::tailZeros(xor_val, 64);
        } else if (c2 == 1) {
            state.elf.previous_lead = 7;
            state.elf.previous_tail = 64;
        } else {
            state.elf.previous_lead = reader.readInt(3);
            int len = (c2 == 2) ? (reader.readInt(4) + 1) : (reader.readInt(6) + 1);
            state.elf.previous_tail = 64 - len - state.elf.previous_lead;
            xor_val = reader.readLong(len) << state.elf.previous_tail;
        }
        
        state.elf.previous_long_value = static_cast<long long>(static_cast<unsigned long long>(state.elf.previous_long_value) ^ static_cast<unsigned long long>(xor_val));
        union { long long ll; double d; } u; u.ll = state.elf.previous_long_value;
        
        if (c1) {
            double vPrime = u.d;
            int sp = Elf64Utils::getSP(std::fabs(vPrime));
            if (betaStar == 0) {
                double res = Elf64Utils::get10iN(-sp - 1);
                return vPrime < 0 ? -res : res;
            } else {
                return Elf64Utils::roundUp(vPrime, betaStar - sp - 1);
            }
        }
        return u.d;
    }

    // --------- Camel ---------
    template <typename StateT>
    static inline void encode_camel(double value, StateT& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::camel;
        long long integer = static_cast<long long>(std::floor(value));
        if (state.camel.first) {
            writer.write(value, 64);
            state.camel.first = false;
        } else {
            long long diff = integer - state.camel.previous_integer;
            if (diff >= -1 && diff <= 1) {
                writer.write(static_cast<long long>(diff + 1), 2);
            } else {
                writer.write(3, 2);
                bool non_negative = diff >= 0;
                writer.write(non_negative);
                long long abs_diff = diff >= 0 ? diff : -diff;
                bool large = abs_diff >= 8;
                writer.write(large);
                writer.write(abs_diff, large ? 16 : 3);
            }

            double dec = value - static_cast<double>(integer);
            int l = CamelTools::decimal_count(value);
            writer.write(l - 1, 2);
            double dxor = dec;
            if (dec >= CamelTools::quick_pow2(-l)) {
                writer.write(true);
                dxor = CamelTools::calculate_dxor(dec, l);
                long long vd_bits = utils::binary_tools::xor_double(1.0 + dec, 1.0 + dxor);
                unsigned long long shifted = static_cast<unsigned long long>(vd_bits) >> (52 - l);
                writer.write(static_cast<long long>(shifted), l);
            } else {
                writer.write(false);
            }

            long long ldxor = static_cast<long long>(std::llround(dxor * CamelTools::quick_pow10(l)));
            if (l == 1) writer.write(ldxor, 3);
            else if (l == 2) {
                bool gt8 = ldxor >= 8; writer.write(gt8); writer.write(ldxor, gt8 ? 5 : 3);
            } else if (l == 3) {
                const int thresholds[3] = {2, 8, 32};
                const int cost_bits[4] = {1, 3, 5, -l + CamelTools::calculate_max(l)};
                int code = 0;
                while (code < 3 && ldxor >= thresholds[code]) code++;
                writer.write(code, 2); writer.write(ldxor, cost_bits[code]);
            } else if (l == 4) {
                const int thresholds[3] = {16, 64, 256};
                const int cost_bits[4] = {4, 6, 8, -l + CamelTools::calculate_max(l)};
                int code = 0;
                while (code < 3 && ldxor >= thresholds[code]) code++;
                writer.write(code, 2); writer.write(ldxor, cost_bits[code]);
            }
        }
        state.camel.previous_integer = integer;
    }

    template <typename StateT>
    static inline double decode_camel(StateT& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::camel;
        if (state.camel.first) {
            double value = reader.readDouble(64);
            state.camel.previous_integer = CamelTools::truncate(value);
            state.camel.first = false;
            return value;
        }

        long long diff = reader.readLong(2);
        long long current_int;
        if (diff <= 2) {
            current_int = state.camel.previous_integer + diff - 1;
        } else {
            long long sign = reader.readBoolean() ? 1 : -1;
            bool gt8 = reader.readBoolean();
            long long magnitude = reader.readLong(gt8 ? 16 : 3);
            current_int = state.camel.previous_integer + sign * magnitude;
        }
        state.camel.previous_integer = current_int;

        int l = reader.readInt(2) + 1;
        long long ldxor = 0;
        bool c1 = reader.readBoolean();
        unsigned long long vd = 0;
        if (c1 && l > 0) vd = static_cast<unsigned long long>(reader.readLong(l)) << (52 - l);

        if (l == 1) ldxor = reader.readLong(3);
        else if (l == 2) {
            bool gt8 = reader.readBoolean(); ldxor = reader.readLong(gt8 ? 5 : 3);
        } else if (l == 3) {
            const int cost_bits[4] = {1, 3, 5, -l + CamelTools::calculate_max(l)};
            int code = reader.readInt(2); ldxor = reader.readLong(cost_bits[code]);
        } else if (l == 4) {
            const int cost_bits[4] = {4, 6, 8, -l + CamelTools::calculate_max(l)};
            int code = reader.readInt(2); ldxor = reader.readLong(cost_bits[code]);
        }

        double dxor = static_cast<double>(ldxor) / CamelTools::quick_pow10(l);
        if (c1) dxor = utils::binary_tools::xor_double(static_cast<long long>(vd), 1.0 + dxor) - 1.0;
        
        return static_cast<double>(current_int) + dxor;
    }

    // --------- DeXORPlus ---------
    static inline int getEndWithEpsilon_dexorplus(double value, int epsilon) {
        using namespace encoding_algorithm::dexor;
        if (DeXORTools::comp(value, 0, DeXORTools::EQUAL_EPS) == 0) return 0;
        int q = epsilon;
        double vq = value / DeXORTools::getP10(q);
        if (DeXORTools::isInt(vq, DeXORTools::INTEGER_EPS)) {
            vq = value / DeXORTools::getP10(q + 1);
            while (DeXORTools::isInt(vq, DeXORTools::INTEGER_EPS)) {
                q++;
                vq = value / DeXORTools::getP10(q + 1);
            }
            return q;
        }
        return epsilon;
    }

    template <typename StateT>
    static inline void encode_dexorplus(double value, StateT& state, utils::MemoryStreamWriter& writer) {
        using namespace encoding_algorithm::dexor;
        double temp = state.dexorplus.previous_value;
        int epsilon = state.dexorplus.epsilon;
        int q = getEndWithEpsilon_dexorplus(value, epsilon);

        int o = std::max(epsilon, q);
        int delta = 0;
        double alpha = 0;
        while (delta < 16) {
            double pow = DeXORTools::getP10(o);
            long long a = DeXORTools::truncate(value / pow);
            long long b = DeXORTools::truncate(temp / pow);
            if (a == b) {
                alpha = a * pow;
                break;
            }
            delta++;
            o++;
        }

        double residual = value - alpha;
        double pow_val;
        long long beta;
        double beta_star;
        if (q <= epsilon) {
            writer.write(true);
            pow_val = DeXORTools::getP10(epsilon);
            beta = DeXORTools::truncate(residual / pow_val);
            beta_star = beta * pow_val;
            delta = o - epsilon;
            if (o == state.dexorplus.p_o) {
                writer.write(true);
            } else {
                writer.write(false);
                state.dexorplus.p_o = o;
                writer.write((long long)delta, 4);
            }
        } else {
            writer.write(false);
            pow_val = DeXORTools::getP10(q);
            beta = DeXORTools::truncate(residual / pow_val);
            beta_star = beta * pow_val;
            delta = o - q;
            if (q == state.dexorplus.p_q) {
                writer.write(true);
            } else {
                writer.write(false);
                state.dexorplus.p_q = q;
                writer.write((long long)(q - epsilon - 1), 4);
            }
            if (delta == state.dexorplus.p_delta) {
                writer.write(true);
            } else {
                writer.write(false);
                state.dexorplus.p_delta = delta;
                writer.write((long long)delta, 4);
            }
        }

        long long abs_beta = std::abs((long long)beta);
        
        if (DeXORTools::comp(alpha, 0) == 0) {
            writer.write(value > 0);
        }

        writer.write(abs_beta, DeXORTools::decimalBits(delta));
        state.dexorplus.previous_value = alpha + beta_star;
    }

    template <typename StateT>
    static inline double decode_dexorplus(StateT& state, utils::MemoryBlockStreamReader& reader) {
        using namespace encoding_algorithm::dexor;
        double temp = state.dexorplus.previous_value;
        int epsilon = state.dexorplus.epsilon;

        double alpha = 0.0;
        int o, q, delta;
        if (reader.readBoolean()) {
            q = epsilon;
            if (reader.readBoolean()) {
                o = state.dexorplus.p_o;
                delta = o - q;
            } else {
                delta = reader.readInt(4);
                o = q + delta;
            }
            state.dexorplus.p_o = o;
        } else {
            if (reader.readBoolean()) {
                q = state.dexorplus.p_q;
            } else {
                q = reader.readInt(4) + epsilon + 1;
            }
            state.dexorplus.p_q = q;
            if (reader.readBoolean()) {
                delta = state.dexorplus.p_delta;
            } else {
                delta = reader.readInt(4);
            }
            state.dexorplus.p_delta = delta;
        }

        o = q + delta;
        double pow_val = DeXORTools::getP10(o);
        alpha = DeXORTools::truncate(temp / pow_val) * pow_val;

        long long sign = alpha > 0 ? 1 : -1;
        if (DeXORTools::comp(alpha, 0) == 0) sign = reader.readBoolean() ? 1 : -1;
        long long beta = sign * reader.readLong(DeXORTools::decimalBits(delta));
        double beta_star = beta * DeXORTools::getP10(q);

        double value = alpha + beta_star;
        state.dexorplus.previous_value = value;
        return value;
    }
};

struct DeXORCodecPolicy {
    using StateType = DeXORState;
    static constexpr AlgorithmType type = AlgorithmType::DeXOR;
    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        CompressedCodec::encode_dexor(value, state, writer);
    }
    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        double out_val = CompressedCodec::decode_dexor(state, reader);
        state.current_value = out_val;
        return out_val;
    }
};

struct GorillaCodecPolicy {
    using StateType = GorillaState;
    static constexpr AlgorithmType type = AlgorithmType::Gorilla;
    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        CompressedCodec::encode_gorilla(value, state, writer);
    }
    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        double out_val = CompressedCodec::decode_gorilla(state, reader);
        state.current_value = out_val;
        return out_val;
    }
};

struct ElfCodecPolicy {
    using StateType = ElfState;
    static constexpr AlgorithmType type = AlgorithmType::Elf;
    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        CompressedCodec::encode_elf(value, state, writer);
    }
    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        double out_val = CompressedCodec::decode_elf(state, reader);
        state.current_value = out_val;
        return out_val;
    }
};

struct CamelCodecPolicy {
    using StateType = CamelState;
    static constexpr AlgorithmType type = AlgorithmType::Camel;
    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        CompressedCodec::encode_camel(value, state, writer);
    }
    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        double out_val = CompressedCodec::decode_camel(state, reader);
        state.current_value = out_val;
        return out_val;
    }
};

struct DeXORPlusCodecPolicy {
    using StateType = DeXORPlusState;
    static constexpr AlgorithmType type = AlgorithmType::DeXORPlus;
    static inline void encode(double value, StateType& state, utils::MemoryStreamWriter& writer) {
        state.current_value = value;
        CompressedCodec::encode_dexorplus(value, state, writer);
    }
    static inline double decode(StateType& state, utils::MemoryBlockStreamReader& reader) {
        double out_val = CompressedCodec::decode_dexorplus(state, reader);
        state.current_value = out_val;
        return out_val;
    }
};

} // namespace codecs
} // namespace hnswlib
