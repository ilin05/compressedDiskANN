#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "lvc_codec/hnswlib/compressed_codecs.h"

namespace diskann::lvc {

inline uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline void append_u32(std::vector<uint8_t> &target, uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) {
        target.push_back(static_cast<uint8_t>(value >> (8 * byte)));
    }
}

inline uint32_t read_u32(const uint8_t *source) {
    return uint32_t(source[0]) | (uint32_t(source[1]) << 8) |
           (uint32_t(source[2]) << 16) | (uint32_t(source[3]) << 24);
}

struct EncodedRecord {
    std::vector<uint8_t> bytes;
    std::vector<float> codec_decoded;
    uint32_t patches = 0;
};

// The codec consumes exact F64 representations of F32 inputs. A patch stores
// original F32 bits only when the codec's canonical output differs.
template <typename Codec>
EncodedRecord encode_record(const float *source, uint32_t dim,
                            const std::vector<typename Codec::StateType> &parent_states) {
    if (parent_states.size() != dim) throw std::runtime_error("Codec state dimension mismatch");
    auto encoder_states = parent_states;
    std::vector<char> codec_bytes;
    utils::MemoryStreamWriter writer(&codec_bytes);
    for (uint32_t j = 0; j < dim; ++j) {
        Codec::encode(static_cast<double>(source[j]), encoder_states[j], writer);
    }
    writer.align();

    auto decoder_states = parent_states;
    utils::MemoryBlockStreamReader reader(
        reinterpret_cast<const unsigned char *>(codec_bytes.data()));
    reader.resetBuffer(reinterpret_cast<const unsigned char *>(codec_bytes.data()), codec_bytes.size());
    EncodedRecord result;
    result.codec_decoded.resize(dim);
    for (uint32_t j = 0; j < dim; ++j) {
        result.codec_decoded[j] = static_cast<float>(Codec::decode(decoder_states[j], reader));
    }

    append_u32(result.bytes, static_cast<uint32_t>(codec_bytes.size()));
    result.bytes.insert(result.bytes.end(), codec_bytes.begin(), codec_bytes.end());
    const size_t patch_count_offset = result.bytes.size();
    append_u32(result.bytes, 0);
    for (uint32_t j = 0; j < dim; ++j) {
        const uint32_t original_bits = bits(source[j]);
        if (original_bits != bits(result.codec_decoded[j])) {
            append_u32(result.bytes, j);
            append_u32(result.bytes, original_bits);
            ++result.patches;
        }
    }
    for (int byte = 0; byte < 4; ++byte) {
        result.bytes[patch_count_offset + byte] =
            static_cast<uint8_t>(result.patches >> (8 * byte));
    }
    return result;
}

// Ancestors update only codec states. Their F32 patches are applied when that
// node itself is requested, keeping parent state identical to codec decoding.
template <typename Codec>
void decode_record(const uint8_t *record, size_t record_size,
                   std::vector<typename Codec::StateType> &states, float *output) {
    if (record_size < 8) throw std::runtime_error("Truncated codec record");
    const uint32_t codec_size = read_u32(record);
    if (uint64_t(codec_size) + 8 > record_size) throw std::runtime_error("Invalid codec record length");
    utils::MemoryBlockStreamReader reader(record + 4);
    reader.resetBuffer(record + 4, codec_size);
    for (size_t j = 0; j < states.size(); ++j) {
        const double value = Codec::decode(states[j], reader);
        if (output) output[j] = static_cast<float>(value);
    }
    const size_t patch_start = 4 + codec_size;
    const uint32_t patch_count = read_u32(record + patch_start);
    if (uint64_t(patch_count) * 8 != record_size - patch_start - 4) {
        throw std::runtime_error("Invalid codec exception patch length");
    }
    if (output) {
        for (uint32_t p = 0; p < patch_count; ++p) {
            const uint32_t dimension = read_u32(record + patch_start + 4 + p * 8);
            const uint32_t original_bits = read_u32(record + patch_start + 8 + p * 8);
            if (dimension >= states.size()) throw std::runtime_error("Patch dimension out of range");
            std::memcpy(output + dimension, &original_bits, sizeof(original_bits));
        }
    }
}

} // namespace diskann::lvc
