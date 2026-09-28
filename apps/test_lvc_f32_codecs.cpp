#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../include/lvc_f32_codec.h"

namespace {

using diskann::lvc::bits;

struct Options {
    std::string input;
    std::string query;
    std::string csv;
    uint32_t count = 4096;
    uint32_t id_start = 0;
    uint32_t chain_length = 4;
    uint64_t seed = 20260928;
};

struct FbinFile {
    std::ifstream stream;
    uint32_t count = 0;
    uint32_t dim = 0;

    explicit FbinFile(const std::string &path) : stream(path, std::ios::binary) {
        if (!stream.read(reinterpret_cast<char *>(&count), sizeof(count)) ||
            !stream.read(reinterpret_cast<char *>(&dim), sizeof(dim)) || count == 0 || dim == 0) {
            throw std::runtime_error("Invalid fbin header: " + path);
        }
        stream.seekg(0, std::ios::end);
        const auto size = stream.tellg();
        const uint64_t expected = 8ULL + uint64_t(count) * dim * sizeof(float);
        if (size < 0 || static_cast<uint64_t>(size) != expected) {
            throw std::runtime_error("Unexpected fbin size: " + path);
        }
    }

    void read(uint32_t id, float *destination) {
        if (id >= count) throw std::runtime_error("fbin vector ID out of range");
        const uint64_t offset = 8ULL + uint64_t(id) * dim * sizeof(float);
        stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!stream.read(reinterpret_cast<char *>(destination), dim * sizeof(float))) {
            throw std::runtime_error("Failed to read fbin vector " + std::to_string(id));
        }
    }
};

float l2(const float *a, const float *b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < dim; ++j) {
        const float diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

Options parse(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value after " + arg);
        const std::string value = argv[++i];
        if (arg == "--input") options.input = value;
        else if (arg == "--query") options.query = value;
        else if (arg == "--csv") options.csv = value;
        else if (arg == "--count") options.count = static_cast<uint32_t>(std::stoul(value));
        else if (arg == "--id-start") options.id_start = static_cast<uint32_t>(std::stoul(value));
        else if (arg == "--chain-length") options.chain_length = static_cast<uint32_t>(std::stoul(value));
        else if (arg == "--seed") options.seed = std::stoull(value);
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (options.input.empty() || options.count == 0 || options.chain_length == 0) {
        throw std::runtime_error("Usage: test_lvc_f32_codecs --input FILE [--query FILE] "
                                 "[--count 4096] [--id-start 0] [--chain-length 4] "
                                 "[--seed 20260928] [--csv FILE]");
    }
    return options;
}

std::vector<uint32_t> choose_ids(uint32_t total, const Options &options, bool random_order) {
    std::vector<uint32_t> ids;
    ids.reserve(options.count);
    if (!random_order) {
        if (uint64_t(options.id_start) + options.count > total) {
            throw std::runtime_error("ID-order sample exceeds fbin vector count");
        }
        for (uint32_t i = 0; i < options.count; ++i) ids.push_back(options.id_start + i);
    } else {
        std::mt19937_64 generator(options.seed);
        std::uniform_int_distribution<uint32_t> distribution(0, total - 1);
        std::unordered_set<uint32_t> chosen;
        while (ids.size() < options.count) {
            const uint32_t id = distribution(generator);
            if (chosen.insert(id).second) ids.push_back(id);
        }
    }
    return ids;
}

struct Result {
    uint64_t correct_vectors = 0;
    uint64_t correct_coordinates = 0;
    uint64_t correct_distances = 0;
    uint64_t payload_bytes = 0;
    uint64_t patches = 0;
    uint64_t codec_correct_vectors = 0;
    uint64_t first_codec_bad_position = UINT64_MAX;
    uint32_t first_codec_bad_dim = 0;
    float first_codec_raw = 0;
    float first_codec_decoded = 0;
    std::vector<float> first_codec_decoded_vector;
    uint64_t first_bad_position = UINT64_MAX;
    uint32_t first_bad_dim = 0;
    float first_raw = 0;
    float first_decoded = 0;
    std::vector<float> first_decoded_vector;
};

template <typename Codec>
Result run_codec(const std::vector<float> &vectors, const std::vector<uint32_t> &ids,
                 const std::vector<float> &queries, uint32_t dim, uint32_t chain_length) {
    const size_t count = ids.size();
    const size_t query_count = queries.size() / dim;
    std::vector<uint8_t> payload;
    std::vector<size_t> offsets(count + 1);
    Result result;

    for (size_t i = 0; i < count; ++i) {
        std::vector<typename Codec::StateType> states(dim);
        const size_t root = (i / chain_length) * chain_length;
        for (size_t ancestor = root; ancestor < i; ++ancestor) {
            diskann::lvc::decode_record<Codec>(payload.data() + offsets[ancestor],
                                               offsets[ancestor + 1] - offsets[ancestor], states, nullptr);
        }
        for (uint32_t j = 0; j < dim; ++j) {
            const float raw = vectors[i * dim + j];
            if (!std::isfinite(raw)) {
                throw std::runtime_error("Non-finite F32 value at vector ID " +
                                         std::to_string(ids[i]) + ", dimension " + std::to_string(j));
            }
        }
        auto record = diskann::lvc::encode_record<Codec>(vectors.data() + i * dim, dim, states);

        offsets[i] = payload.size();
        for (uint32_t j = 0; j < dim; ++j) {
            const uint32_t original_bits = bits(vectors[i * dim + j]);
            if (original_bits != bits(record.codec_decoded[j])) {
                if (result.first_codec_bad_position == UINT64_MAX) {
                    result.first_codec_bad_position = i;
                    result.first_codec_bad_dim = j;
                    result.first_codec_raw = vectors[i * dim + j];
                    result.first_codec_decoded = record.codec_decoded[j];
                    result.first_codec_decoded_vector = record.codec_decoded;
                }
            }
        }
        payload.insert(payload.end(), record.bytes.begin(), record.bytes.end());
        if (record.patches == 0) ++result.codec_correct_vectors;
        result.patches += record.patches;
        offsets[i + 1] = payload.size();
    }

    result.payload_bytes = payload.size();
    std::vector<float> decoded(dim);
    for (size_t i = 0; i < count; ++i) {
        std::vector<typename Codec::StateType> states(dim);
        const size_t root = (i / chain_length) * chain_length;
        for (size_t node = root; node <= i; ++node) {
            diskann::lvc::decode_record<Codec>(payload.data() + offsets[node],
                                               offsets[node + 1] - offsets[node], states,
                                               node == i ? decoded.data() : nullptr);
        }

        bool whole_vector_correct = true;
        for (uint32_t j = 0; j < dim; ++j) {
            const float raw = vectors[i * dim + j];
            if (bits(raw) == bits(decoded[j])) {
                ++result.correct_coordinates;
            } else {
                whole_vector_correct = false;
                if (result.first_bad_position == UINT64_MAX) {
                    result.first_bad_position = i;
                    result.first_bad_dim = j;
                    result.first_raw = raw;
                    result.first_decoded = decoded[j];
                    result.first_decoded_vector = decoded;
                }
            }
        }
        if (whole_vector_correct) ++result.correct_vectors;
        for (size_t q = 0; q < query_count; ++q) {
            const float raw_distance = l2(vectors.data() + i * dim, queries.data() + q * dim, dim);
            const float decoded_distance = l2(decoded.data(), queries.data() + q * dim, dim);
            if (bits(raw_distance) == bits(decoded_distance)) ++result.correct_distances;
        }
    }
    return result;
}

template <typename Codec>
bool report(const char *codec_name, const char *order_name, const Result &result,
            const std::vector<uint32_t> &ids, uint32_t dim, size_t query_count,
            uint32_t chain_length, std::ofstream &csv) {
    const uint64_t raw_bytes = uint64_t(ids.size()) * dim * sizeof(float);
    const uint64_t metadata_bytes = ids.size() * (sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t));
    const double payload_ratio = double(raw_bytes) / result.payload_bytes;
    const double total_ratio = double(raw_bytes) / (result.payload_bytes + metadata_bytes);
    const uint64_t total_coordinates = uint64_t(ids.size()) * dim;
    const uint64_t total_distances = uint64_t(ids.size()) * query_count;
    std::cout << codec_name << " " << order_name
              << " vectors=" << result.correct_vectors << "/" << ids.size()
              << " coordinates=" << result.correct_coordinates << "/" << total_coordinates
              << " L2=" << result.correct_distances << "/" << total_distances
              << " codec_vectors_before_patch=" << result.codec_correct_vectors << "/" << ids.size()
              << " patches=" << result.patches
              << " payload_bytes=" << result.payload_bytes
              << " raw_F32_bytes=" << raw_bytes
              << " metadata_bytes=" << metadata_bytes
              << " payload_ratio=" << std::fixed << std::setprecision(4) << payload_ratio
              << " total_ratio=" << total_ratio << '\n';
    if (csv) {
        csv << codec_name << ',' << order_name << ',' << ids.size() << ',' << dim << ',' << chain_length
            << ',' << result.correct_vectors << ',' << result.correct_coordinates
            << ',' << result.correct_distances << ',' << total_distances
            << ',' << result.codec_correct_vectors << ',' << result.patches
            << ',' << raw_bytes << ',' << result.payload_bytes << ',' << metadata_bytes
            << ',' << payload_ratio << ',' << total_ratio << '\n';
    }
    if (result.first_codec_bad_position != UINT64_MAX) {
        std::cout << "FIRST_CODEC_MISMATCH_BEFORE_PATCH sample_position="
                  << result.first_codec_bad_position
                  << " vector_id=" << ids[result.first_codec_bad_position]
                  << " dimension=" << result.first_codec_bad_dim
                  << " original=" << std::setprecision(9) << result.first_codec_raw
                  << " decoded=" << result.first_codec_decoded
                  << " original_bits=0x" << std::hex << bits(result.first_codec_raw)
                  << " decoded_bits=0x" << bits(result.first_codec_decoded) << std::dec << '\n';
        std::cout << "actual_decoded_vector_before_patch=[";
        for (size_t j = 0; j < result.first_codec_decoded_vector.size(); ++j) {
            if (j) std::cout << ',';
            std::cout << result.first_codec_decoded_vector[j];
        }
        std::cout << "]\n";
    }
    if (result.first_bad_position != UINT64_MAX) {
        std::cout << "FIRST_MISMATCH sample_position=" << result.first_bad_position
                  << " vector_id=" << ids[result.first_bad_position]
                  << " dimension=" << result.first_bad_dim
                  << " original=" << std::setprecision(9) << result.first_raw
                  << " decoded=" << result.first_decoded
                  << " original_bits=0x" << std::hex << bits(result.first_raw)
                  << " decoded_bits=0x" << bits(result.first_decoded) << std::dec << '\n';
        std::cout << "actual_decoded_vector=[";
        for (size_t j = 0; j < result.first_decoded_vector.size(); ++j) {
            if (j) std::cout << ',';
            std::cout << result.first_decoded_vector[j];
        }
        std::cout << "]\n";
    }
    return result.correct_vectors == ids.size() && result.correct_distances == total_distances;
}

template <typename Codec>
bool run_and_report(const char *codec_name, const char *order_name,
                    const std::vector<float> &vectors, const std::vector<uint32_t> &ids,
                    const std::vector<float> &queries, uint32_t dim, uint32_t chain_length,
                    std::ofstream &csv) {
    return report<Codec>(codec_name, order_name,
                         run_codec<Codec>(vectors, ids, queries, dim, chain_length),
                         ids, dim, queries.size() / dim, chain_length, csv);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse(argc, argv);
        FbinFile input(options.input);
        if (options.count > input.count) throw std::runtime_error("Sample count exceeds dataset size");

        std::vector<float> queries;
        if (!options.query.empty()) {
            FbinFile query(options.query);
            if (query.dim != input.dim) throw std::runtime_error("Query dimension does not match dataset");
            const uint32_t count = std::min<uint32_t>(4, query.count);
            queries.resize(size_t(count) * input.dim);
            for (uint32_t i = 0; i < count; ++i) query.read(i, queries.data() + size_t(i) * input.dim);
        }

        std::ofstream csv;
        if (!options.csv.empty()) {
            csv.open(options.csv);
            if (!csv) throw std::runtime_error("Cannot open CSV: " + options.csv);
            csv << "codec,order,vectors,dimension,chain_length,correct_vectors,correct_coordinates,"
                   "correct_l2,total_l2,codec_correct_vectors_before_patch,patches,"
                   "raw_f32_bytes,payload_bytes,metadata_bytes,payload_ratio,total_ratio\n";
        }

        bool all_correct = true;
        for (bool random_order : {false, true}) {
            const char *order_name = random_order ? "random" : "id";
            const auto ids = choose_ids(input.count, options, random_order);
            std::vector<float> vectors(size_t(ids.size()) * input.dim);
            for (size_t i = 0; i < ids.size(); ++i) {
                input.read(ids[i], vectors.data() + i * input.dim);
            }
            if (queries.empty()) {
                queries.assign(vectors.begin(), vectors.begin() + size_t(std::min<size_t>(4, ids.size())) * input.dim);
            }
            std::cout << "dataset=" << options.input << " order=" << order_name
                      << " total_vectors=" << input.count << " sample_vectors=" << ids.size()
                      << " dim=" << input.dim << " chain_length=" << options.chain_length
                      << " seed=" << options.seed << '\n';
            all_correct = run_and_report<hnswlib::codecs::DeXORCodecPolicy>(
                              "DeXOR", order_name, vectors, ids, queries, input.dim, options.chain_length, csv) && all_correct;
            all_correct = run_and_report<hnswlib::codecs::GorillaCodecPolicy>(
                              "Gorilla", order_name, vectors, ids, queries, input.dim, options.chain_length, csv) && all_correct;
            all_correct = run_and_report<hnswlib::codecs::ElfCodecPolicy>(
                              "Elf", order_name, vectors, ids, queries, input.dim, options.chain_length, csv) && all_correct;
        }
        return all_correct ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << "D0 test failed: " << error.what() << '\n';
        return 1;
    }
}
