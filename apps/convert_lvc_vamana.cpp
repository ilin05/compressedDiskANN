#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
                   const std::vector<float> &vectors, uint32_t count, uint32_t dim,
                   bool compression_only, std::ofstream &stats, double mean_reference_l2,
                   uint32_t max_depth) {
    const auto start = std::chrono::steady_clock::now();
    const auto vector = [&](uint32_t id) { return vectors.data() + uint64_t(id) * dim; };
    auto store = diskann::lvc::RecordStore<Codec>::build(forest, dim, vector);
    const std::string prefix = output + "_" + name;
    if (compression_only) {
        std::vector<float> decoded(dim);
        for (uint32_t id = 0; id < count; ++id) {
            store.decode(id, decoded.data());
            if (std::memcmp(decoded.data(), vector(id), dim * sizeof(float)) != 0)
                throw std::runtime_error(std::string(name) + " lossless decode failed at node " + std::to_string(id));
        }
    } else {
        std::filesystem::copy_file(source, prefix, std::filesystem::copy_options::overwrite_existing);
        diskann::lvc::write_record_file(prefix + ".data", count, dim, mode, store);
    }
    const uint64_t raw_bytes = uint64_t(count) * dim * sizeof(float);
    const uint64_t metadata_bytes = compression_only
        ? 8 + 8 + 4 + 4 + uint64_t(count) * (sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2) + 8
        : std::filesystem::file_size(prefix + ".data") - store.bytes.size();
    const uint64_t graph_bytes = std::filesystem::file_size(source);
    const uint64_t raw_index_bytes = graph_bytes + 32 + raw_bytes;
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
    stats << name << ',' << count << ',' << forest.initial_roots << ',' << forest.fallback_nodes << ','
          << max_depth << ',' << mean_reference_l2 << ',' << store.patches << ',' << store.patches * 8 << ',' << store.bytes.size()
          << ',' << metadata_bytes << ',' << graph_bytes << ',' << raw_bytes << ','
          << double(raw_bytes) / store.bytes.size() << ','
          << double(raw_index_bytes) / (graph_bytes + store.bytes.size() + metadata_bytes) << ','
          << (compression_only ? 0 : 1) << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 4) throw std::runtime_error(
            "Usage: convert_lvc_vamana GRAPH_PREFIX VECTORS.fbin OUTPUT_PREFIX [--chain_max 2|4|8|-1] [--compression_only]");
        const std::string source = argv[1];
        const std::string input = argv[2];
        const std::string output = argv[3];
        int chain_max = 2;
        bool compression_only = false;
        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--chain_max" && i + 1 < argc) chain_max = std::stoi(argv[++i]);
            else if (arg == "--compression_only") compression_only = true;
            else throw std::runtime_error("Unknown conversion option: " + arg);
        }
        if (chain_max != 2 && !compression_only)
            throw std::runtime_error("Deep chains require --compression_only; searchable Vamana format is chain=2");
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
        const auto forest = diskann::lvc::build_forest(count, dim, neighbors, vector, chain_max);
        uint32_t max_depth = 0;
        double distance_sum = 0;
        std::ofstream forest_csv(output + "_forest.csv");
        forest_csv << "node,degree,parent,depth,reference_l2\n";
        for (uint32_t id = 0; id < count; ++id) {
            max_depth = std::max(max_depth, forest.depth[id]);
            double distance = 0;
            if (forest.parent[id] != diskann::lvc::no_parent) {
                const float *a = vector(id), *b = vector(forest.parent[id]);
                for (uint32_t j = 0; j < dim; ++j) {
                    const double delta = double(a[j]) - double(b[j]);
                    distance += delta * delta;
                }
                distance = std::sqrt(distance);
                distance_sum += distance;
            }
            forest_csv << id << ',' << graph[id].size() << ',';
            if (forest.parent[id] != diskann::lvc::no_parent) forest_csv << forest.parent[id];
            forest_csv << ',' << forest.depth[id] << ',' << distance << '\n';
        }
        const double mean_reference_l2 = distance_sum / (count - forest.initial_roots == 0 ? 1 : count - forest.initial_roots);
        std::ofstream stats(output + "_compression.csv");
        stats << "codec,nodes,roots,fallback,max_depth,mean_reference_l2,patches,patch_bytes,payload_bytes,metadata_bytes,graph_bytes,raw_f32_bytes,payload_ratio_f32,full_index_ratio_f32,record_file_materialized\n";
        std::cout << "forest nodes=" << count << " dim=" << dim
                  << " roots=" << forest.initial_roots
                  << " fallback=" << forest.fallback_nodes
                  << " rounds=" << forest.propagation_rounds
                  << " max_depth=" << max_depth << " mean_reference_l2=" << mean_reference_l2 << '\n';

        const std::string raw_prefix = output + "_raw";
        if (!compression_only) {
            std::filesystem::copy_file(source, raw_prefix, std::filesystem::copy_options::overwrite_existing);
            diskann::lvc::write_raw_file(raw_prefix + ".data", count, dim, vectors);
            std::cout << "raw graph_bytes=" << std::filesystem::file_size(raw_prefix)
                      << " data_bytes=" << std::filesystem::file_size(raw_prefix + ".data") << '\n';
        }
        convert_codec<hnswlib::codecs::DeXORCodecPolicy>("dexor", 2, source, output, forest, vectors, count, dim, compression_only, stats, mean_reference_l2, max_depth);
        convert_codec<hnswlib::codecs::GorillaCodecPolicy>("gorilla", 3, source, output, forest, vectors, count, dim, compression_only, stats, mean_reference_l2, max_depth);
        convert_codec<hnswlib::codecs::ElfCodecPolicy>("elf", 4, source, output, forest, vectors, count, dim, compression_only, stats, mean_reference_l2, max_depth);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "D2 conversion failed: " << error.what() << '\n';
        return 1;
    }
}
