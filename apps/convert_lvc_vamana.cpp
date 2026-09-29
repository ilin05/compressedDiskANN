#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/lvc_vamana_file.h"

namespace {

template <typename T> T read(std::ifstream &in) {
    T value{};
    if (!in.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error("Truncated Vamana graph or fbin");
    return value;
}

template <typename Codec>
void convert_codec(const char *name, uint32_t mode, const std::string &source,
                   const std::string &output, const diskann::lvc::Forest &forest,
                   const std::vector<float> &vectors, uint32_t count, uint32_t dim) {
    const auto start = std::chrono::steady_clock::now();
    const auto vector = [&](uint32_t id) { return vectors.data() + uint64_t(id) * dim; };
    auto store = diskann::lvc::RecordStore<Codec>::build(forest, dim, vector);
    const std::string prefix = output + "_" + name;
    std::filesystem::copy_file(source, prefix, std::filesystem::copy_options::overwrite_existing);
    diskann::lvc::write_record_file(prefix + ".data", count, dim, mode, store);
    const uint64_t raw_bytes = uint64_t(count) * dim * sizeof(float);
    const uint64_t metadata_bytes = std::filesystem::file_size(prefix + ".data") - store.bytes.size();
    const uint64_t graph_bytes = std::filesystem::file_size(prefix);
    const uint64_t raw_index_bytes = std::filesystem::file_size(output + "_raw") +
                                     std::filesystem::file_size(output + "_raw.data");
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << name << " roots=" << forest.initial_roots
              << " fallback=" << forest.fallback_nodes
              << " patches=" << store.patches
              << " payload_bytes=" << store.bytes.size()
              << " metadata_bytes=" << metadata_bytes
              << " graph_bytes=" << graph_bytes
              << " payload_ratio=" << std::fixed << std::setprecision(4)
              << double(raw_bytes) / store.bytes.size()
              << " data_file_ratio=" << double(raw_bytes) / (store.bytes.size() + metadata_bytes)
              << " full_index_ratio=" << double(raw_index_bytes) / (graph_bytes + store.bytes.size() + metadata_bytes)
              << " build_seconds=" << elapsed << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error(
            "Usage: convert_lvc_vamana GRAPH_PREFIX VECTORS.fbin OUTPUT_PREFIX");
        const std::string source = argv[1];
        const std::string input = argv[2];
        const std::string output = argv[3];
        std::ifstream fbin(input, std::ios::binary);
        const uint32_t count = read<uint32_t>(fbin);
        const uint32_t dim = read<uint32_t>(fbin);
        if (count == 0 || dim == 0) throw std::runtime_error("Empty F32 input");
        if (std::filesystem::file_size(input) != 8ULL + uint64_t(count) * dim * sizeof(float))
            throw std::runtime_error("Unexpected fbin size");
        std::vector<float> vectors(uint64_t(count) * dim);
        if (!fbin.read(reinterpret_cast<char *>(vectors.data()), vectors.size() * sizeof(float)))
            throw std::runtime_error("Cannot read F32 input");

        std::ifstream graph_file(source, std::ios::binary);
        const uint64_t graph_size = read<uint64_t>(graph_file);
        (void)read<uint32_t>(graph_file); // max degree
        (void)read<uint32_t>(graph_file); // entry point
        const uint64_t frozen_points = read<uint64_t>(graph_file);
        if (frozen_points != 0 || graph_size != std::filesystem::file_size(source))
            throw std::runtime_error("D2 requires a complete static Vamana graph without frozen points");
        std::vector<std::vector<uint32_t>> graph(count);
        for (uint32_t id = 0; id < count; ++id) {
            const uint32_t degree = read<uint32_t>(graph_file);
            if (degree > count) throw std::runtime_error("Invalid Vamana degree");
            graph[id].resize(degree);
            if (!graph_file.read(reinterpret_cast<char *>(graph[id].data()), uint64_t(degree) * sizeof(uint32_t)))
                throw std::runtime_error("Truncated Vamana neighbors");
        }
        if (static_cast<uint64_t>(graph_file.tellg()) != graph_size)
            throw std::runtime_error("Vamana node count does not match fbin");

        const auto neighbors = [&](uint32_t id) -> const std::vector<uint32_t> & { return graph[id]; };
        const auto vector = [&](uint32_t id) { return vectors.data() + uint64_t(id) * dim; };
        const auto forest = diskann::lvc::build_forest(count, dim, neighbors, vector);
        std::cout << "forest nodes=" << count << " dim=" << dim
                  << " roots=" << forest.initial_roots
                  << " fallback=" << forest.fallback_nodes
                  << " rounds=" << forest.propagation_rounds << '\n';

        const std::string raw_prefix = output + "_raw";
        std::filesystem::copy_file(source, raw_prefix, std::filesystem::copy_options::overwrite_existing);
        diskann::lvc::write_raw_file(raw_prefix + ".data", count, dim, vectors);
        std::cout << "raw graph_bytes=" << std::filesystem::file_size(raw_prefix)
                  << " data_bytes=" << std::filesystem::file_size(raw_prefix + ".data") << '\n';
        convert_codec<hnswlib::codecs::DeXORCodecPolicy>("dexor", 2, source, output, forest, vectors, count, dim);
        convert_codec<hnswlib::codecs::GorillaCodecPolicy>("gorilla", 3, source, output, forest, vectors, count, dim);
        convert_codec<hnswlib::codecs::ElfCodecPolicy>("elf", 4, source, output, forest, vectors, count, dim);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "D2 conversion failed: " << error.what() << '\n';
        return 1;
    }
}
