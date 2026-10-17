// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>
// Verbatim ALP encoder/decoder from apply_differential src/pq_flash_index.cpp.
// Source commit: 4688f833fbbbd377eed928368756857d295802ca
// Retains exponent search, 1e-4 tolerance and bit layout.
namespace diskann { namespace alp_only {
    struct FlashBitWriter {
        std::vector<uint8_t>& buf;
        uint8_t cur_byte = 0;
        int bit_pos = 0;

        FlashBitWriter(std::vector<uint8_t>& b) : buf(b) {}
        void write(uint64_t val, int bits) {
            for (int i = 0; i < bits; ++i) {
                uint8_t b = (val >> i) & 1;
                cur_byte |= (b << bit_pos);
                bit_pos++;
                if (bit_pos == 8) {
                    buf.push_back(cur_byte);
                    cur_byte = 0;
                    bit_pos = 0;
                }
            }
        }
        void flush() {
            if (bit_pos > 0) {
                buf.push_back(cur_byte);
                cur_byte = 0;
                bit_pos = 0;
            }
        }
    };

    struct FlashBitReader {
        const uint8_t* buf;
        size_t byte_pos = 0;
        int bit_pos = 0;

        FlashBitReader(const uint8_t* b) : buf(b) {}
        uint64_t read(int bits) {
            uint64_t val = 0;
            for (int i = 0; i < bits; ++i) {
                uint64_t b = (buf[byte_pos] >> bit_pos) & 1;
                val |= (b << i);
                bit_pos++;
                if (bit_pos == 8) {
                    byte_pos++;
                    bit_pos = 0;
                }
            }
            return val;
        }
    };

    inline int get_flash_bit_width(uint64_t n) {
        if (n == 0) return 0;
        int bits = 0;
        while (n > 0) {
            n >>= 1;
            bits++;
        }
        return bits;
    }

    template <typename T>
    void flash_encode_and_store(const T *uncompressed_vec, size_t dim, std::vector<uint8_t>& out_temp)
    {
        FlashBitWriter writer(out_temp);

        if constexpr (std::is_same_v<T, float>) {
            int selected_exp = 16;
            for (int exp = 0; exp <= 16; ++exp) {
                double factor = std::pow(10.0, exp);
                double inv_factor = 1.0 / factor;
                bool ok = true;
                for (size_t i = 0; i < dim; ++i) {
                    double origin = uncompressed_vec[i];
                    double reconstructed = std::round(origin * factor) * inv_factor;
                    if (std::abs(origin - reconstructed) > 1e-4) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    selected_exp = exp;
                    break;
                }
            }
            writer.write((uint64_t)selected_exp, 8);
            double factor = std::pow(10.0, selected_exp);
            int64_t min_val = INT64_MAX;
            int64_t max_val = INT64_MIN;
            for (size_t i = 0; i < dim; ++i) {
                int64_t val = (int64_t)std::llround(uncompressed_vec[i] * factor);
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            uint64_t range = (uint64_t)(max_val - min_val);
            int bit_width = get_flash_bit_width(range);
            writer.write((uint64_t)bit_width, 8);
            writer.write((uint64_t)min_val, 64);
            for (size_t i = 0; i < dim; ++i) {
                int64_t val = (int64_t)std::llround(uncompressed_vec[i] * factor);
                writer.write((uint64_t)(val - min_val), bit_width);
            }
        } else {
            int64_t min_val = INT64_MAX;
            int64_t max_val = INT64_MIN;
            for (size_t i = 0; i < dim; ++i) {
                int64_t val = (int64_t)uncompressed_vec[i];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            uint64_t range = (uint64_t)(max_val - min_val);
            int bit_width = get_flash_bit_width(range);
            writer.write((uint64_t)bit_width, 8);
            writer.write((uint64_t)min_val, 64);
            for (size_t i = 0; i < dim; ++i) {
                int64_t val = (int64_t)uncompressed_vec[i];
                writer.write((uint64_t)(val - min_val), bit_width);
            }
        }
        writer.flush();
    }

    template <typename T>
    void flash_decode_vector(const uint8_t* compressed_ptr, size_t dim, size_t aligned_dim, T *out_vec)
    {
        FlashBitReader reader(compressed_ptr);

        if constexpr (std::is_same_v<T, float>) {
            int exp = reader.read(8);
            int bit_width = reader.read(8);
            int64_t min_val = (int64_t)reader.read(64);
            double factor = std::pow(10.0, -exp);
            for (size_t i = 0; i < dim; ++i) {
                int64_t delta = (int64_t)reader.read(bit_width);
                int64_t val = min_val + delta;
                out_vec[i] = static_cast<float>(val * factor);
            }
            // Zero out the remaining aligned dimensions
            for (size_t i = dim; i < aligned_dim; ++i) {
                out_vec[i] = 0;
            }
        } else {
            int bit_width = reader.read(8);
            int64_t min_val = (int64_t)reader.read(64);
            for (size_t i = 0; i < dim; ++i) {
                int64_t delta = (int64_t)reader.read(bit_width);
                int64_t val = min_val + delta;
                out_vec[i] = static_cast<T>(val);
            }
            for (size_t i = dim; i < aligned_dim; ++i) {
                out_vec[i] = 0;
            }
        }
    }
}} // namespace diskann::alp_only
