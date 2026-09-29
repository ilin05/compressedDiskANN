#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/lvc_vamana_file.h"

namespace {

template <typename T> T read(std::ifstream &in) {
    T value{};
    if (!in.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error("Truncated LVC data file");
    return value;
}

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float l2(const float *a, const float *b, uint32_t dim) {
    float sum = 0;
    for (uint32_t j = 0; j < dim; ++j) {
        const float delta = a[j] - b[j];
        sum += delta * delta;
    }
    return sum;
}

template <typename Codec>
void verify_codec(uint32_t count, uint32_t dim, diskann::lvc::Forest forest,
                  std::vector<uint64_t> offsets, std::vector<uint64_t> lengths,
                  std::vector<uint8_t> payload, const std::vector<float> &original) {
    diskann::lvc::RecordStore<Codec> store;
    store.forest = std::move(forest);
    store.dim = dim;
    store.offsets = std::move(offsets);
    store.lengths = std::move(lengths);
    store.bytes = std::move(payload);
    std::vector<float> decoded(dim);
    for (uint32_t id = 0; id < count; ++id) {
        store.decode(id, decoded.data());
        const float *raw = original.data() + uint64_t(id) * dim;
        for (uint32_t j = 0; j < dim; ++j) {
            if (bits(decoded[j]) != bits(raw[j])) {
                std::cerr << "FIRST_MISMATCH vector_id=" << id << " dimension=" << j
                          << " original=" << raw[j] << " decoded=" << decoded[j] << '\n';
                std::cerr << "actual_decoded_vector=[";
                for (uint32_t k = 0; k < dim; ++k) {
                    if (k) std::cerr << ',';
                    std::cerr << decoded[k];
                }
                std::cerr << "]\n";
                throw std::runtime_error("F32 bitwise recovery failed");
            }
        }
        for (uint32_t query = 0; query < std::min<uint32_t>(4, count); ++query) {
            const float *q = original.data() + uint64_t(query) * dim;
            if (bits(l2(decoded.data(), q, dim)) != bits(l2(raw, q, dim)))
                throw std::runtime_error("L2 mismatch at vector " + std::to_string(id));
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: verify_lvc_vamana VECTORS.fbin INDEX.data");
        std::ifstream input(argv[1], std::ios::binary);
        const uint32_t count = read<uint32_t>(input);
        const uint32_t dim = read<uint32_t>(input);
        std::vector<float> original(uint64_t(count) * dim);
        if (!input.read(reinterpret_cast<char *>(original.data()), original.size() * sizeof(float)))
            throw std::runtime_error("Cannot read F32 vectors");

        std::ifstream file(argv[2], std::ios::binary);
        if (read<uint32_t>(file) != count || read<uint32_t>(file) != dim ||
            read<uint64_t>(file) != diskann::lvc::vamana_magic ||
            read<uint32_t>(file) != diskann::lvc::vamana_version)
            throw std::runtime_error("LVC data header mismatch");
        const uint32_t mode = read<uint32_t>(file);
        diskann::lvc::Forest forest;
        if (mode != 1) {
            forest.parent.resize(count);
            forest.depth.resize(count);
            file.read(reinterpret_cast<char *>(forest.parent.data()), count * sizeof(uint32_t));
            file.read(reinterpret_cast<char *>(forest.depth.data()), count);
            forest.initial_roots = std::count(forest.parent.begin(), forest.parent.end(), diskann::lvc::no_parent);
            diskann::lvc::validate_forest(forest);
        }
        std::vector<uint64_t> offsets;
        std::vector<uint64_t> lengths;
        if (mode != 1) {
            offsets.resize(count);
            lengths.resize(count);
            file.read(reinterpret_cast<char *>(offsets.data()), count * sizeof(uint64_t));
            file.read(reinterpret_cast<char *>(lengths.data()), count * sizeof(uint64_t));
        }
        const uint64_t payload_size = read<uint64_t>(file);
        std::vector<uint8_t> payload(payload_size);
        if (!file.read(reinterpret_cast<char *>(payload.data()), payload_size))
            throw std::runtime_error("Truncated LVC payload");
        if (mode == 1) {
            if (payload_size != original.size() * sizeof(float) ||
                std::memcmp(payload.data(), original.data(), payload_size) != 0)
                throw std::runtime_error("Raw F32 payload mismatch");
        } else if (mode == 2) {
            verify_codec<hnswlib::codecs::DeXORCodecPolicy>(count, dim, std::move(forest),
                std::move(offsets), std::move(lengths), std::move(payload), original);
        } else if (mode == 3) {
            verify_codec<hnswlib::codecs::GorillaCodecPolicy>(count, dim, std::move(forest),
                std::move(offsets), std::move(lengths), std::move(payload), original);
        } else if (mode == 4) {
            verify_codec<hnswlib::codecs::ElfCodecPolicy>(count, dim, std::move(forest),
                std::move(offsets), std::move(lengths), std::move(payload), original);
        } else {
            throw std::runtime_error("Unknown LVC codec");
        }
        std::cout << "PASS mode=" << mode << " vectors=" << count << '/' << count
                  << " coordinates=" << uint64_t(count) * dim << '/' << uint64_t(count) * dim
                  << " L2=" << uint64_t(count) * std::min<uint32_t>(4, count) << '/'
                  << uint64_t(count) * std::min<uint32_t>(4, count) << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "D2 verification failed: " << error.what() << '\n';
        return 1;
    }
}
