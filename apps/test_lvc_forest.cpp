#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/lvc_forest.h"

namespace {

struct Input {
    uint32_t count = 256;
    uint32_t dim = 16;
    std::vector<float> values;
};

Input load_input(int argc, char **argv) {
    Input input;
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing argument after " + arg);
        const std::string value = argv[++i];
        if (arg == "--input") path = value;
        else if (arg == "--count") input.count = static_cast<uint32_t>(std::stoul(value));
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    if (input.count < 200) throw std::runtime_error("D1 fixture requires at least 200 vectors");
    if (!path.empty()) {
        std::ifstream file(path, std::ios::binary);
        uint32_t total = 0;
        if (!file.read(reinterpret_cast<char *>(&total), 4) ||
            !file.read(reinterpret_cast<char *>(&input.dim), 4) ||
            total < input.count || input.dim == 0) {
            throw std::runtime_error("Invalid or short fbin: " + path);
        }
        input.values.resize(size_t(input.count) * input.dim);
        if (!file.read(reinterpret_cast<char *>(input.values.data()),
                       input.values.size() * sizeof(float))) {
            throw std::runtime_error("Cannot read fbin vectors: " + path);
        }
    } else {
        input.values.resize(size_t(input.count) * input.dim);
        for (uint32_t i = 0; i < input.count; ++i) {
            for (uint32_t j = 0; j < input.dim; ++j) {
                const int32_t numerator = static_cast<int32_t>((i * 7919U + j * 1049U) % 200003U) - 100001;
                input.values[size_t(i) * input.dim + j] = static_cast<float>(numerator) / 3701.0f;
            }
        }
    }
    for (float value : input.values) {
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite input coordinate");
    }
    return input;
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
void test_codec(const char *name, const diskann::lvc::Forest &forest, const Input &input,
                const std::vector<std::vector<uint32_t>> &graph) {
    const auto vector = [&](uint32_t id) { return input.values.data() + size_t(id) * input.dim; };
    auto store = diskann::lvc::RecordStore<Codec>::build(forest, input.dim, vector);
    std::vector<float> decoded(input.dim);
    diskann::lvc::RootStateCache<Codec> state_cache;
    state_cache.reserve(forest.initial_roots);
    for (uint32_t id = 0; id < input.count; ++id)
        if (forest.parent[id] == diskann::lvc::no_parent)
            state_cache.insert(id, store.bytes.data() + store.offsets[id], store.lengths[id], input.dim);
    if (state_cache.size() != forest.initial_roots)
        throw std::runtime_error("State cache root count mismatch");
    diskann::lvc::RootStateCache<Codec> partial_cache;
    const uint32_t cached_root = [&] {
        for (uint32_t id = 0; id < input.count; ++id)
            if (forest.parent[id] == diskann::lvc::no_parent) return id;
        throw std::runtime_error("No root for state cache test");
    }();
    partial_cache.insert(cached_root, store.bytes.data() + store.offsets[cached_root],
                         store.lengths[cached_root], input.dim);
    std::vector<typename Codec::StateType> states;
    std::vector<float> cached(input.dim);
    uint64_t correct_coordinates = 0;
    uint64_t correct_l2 = 0;
    uint32_t non_neighbor_parents = 0;
    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    intervals.reserve(input.count);
    for (uint32_t id = 0; id < input.count; ++id) {
        const uint64_t offset = store.offsets[id];
        const uint64_t length = store.lengths[id];
        if (length < 8 || offset > store.bytes.size() || length > store.bytes.size() - offset) {
            throw std::runtime_error("Invalid record bounds for node " + std::to_string(id));
        }
        intervals.emplace_back(offset, offset + length);
        const uint32_t parent = forest.parent[id];
        if (parent != diskann::lvc::no_parent &&
            std::find(graph[id].begin(), graph[id].end(), parent) == graph[id].end()) {
            ++non_neighbor_parents;
        }
        store.decode(id, decoded.data());
        uint64_t records = 0;
        store.decode(id, cached.data(), states, &state_cache, &records);
        const uint64_t expected_records = parent == diskann::lvc::no_parent ? 0 : 1;
        if (records != expected_records || std::memcmp(decoded.data(), cached.data(),
                                                        input.dim * sizeof(float)) != 0)
            throw std::runtime_error(std::string(name) + " state cache mismatch at node " + std::to_string(id));
        store.decode(id, cached.data(), states, &partial_cache, &records);
        const uint64_t partial_records = (parent == diskann::lvc::no_parent ? 1 : 2) -
                                         (id == cached_root || parent == cached_root ? 1 : 0);
        if (records != partial_records || std::memcmp(decoded.data(), cached.data(),
                                                     input.dim * sizeof(float)) != 0)
            throw std::runtime_error(std::string(name) + " partial state cache mismatch at node " + std::to_string(id));
        for (uint32_t j = 0; j < input.dim; ++j) {
            if (diskann::lvc::bits(decoded[j]) != diskann::lvc::bits(vector(id)[j])) {
                std::cerr << "FIRST_MISMATCH codec=" << name << " vector_id=" << id
                  << " dimension=" << j << " original=" << std::setprecision(9) << vector(id)[j]
                  << " decoded=" << decoded[j] << '\n';
                std::cerr << "actual_decoded_vector=[";
                for (uint32_t k = 0; k < input.dim; ++k) {
                    if (k) std::cerr << ',';
                    std::cerr << decoded[k];
                }
                std::cerr << "]\n";
                throw std::runtime_error("F32 bitwise recovery failed");
            }
            ++correct_coordinates;
        }
        for (uint32_t q : {0U, 1U, 100U, 199U}) {
            if (diskann::lvc::bits(l2(decoded.data(), vector(q), input.dim)) !=
                diskann::lvc::bits(l2(vector(id), vector(q), input.dim))) {
                throw std::runtime_error(std::string(name) + " L2 mismatch at node " + std::to_string(id));
            }
            ++correct_l2;
        }
    }
    std::sort(intervals.begin(), intervals.end());
    uint64_t end = 0;
    for (auto [start, stop] : intervals) {
        if (start != end) throw std::runtime_error("Gap or overlap in LVC records");
        end = stop;
    }
    if (end != store.bytes.size() || non_neighbor_parents == 0) {
        throw std::runtime_error("Incomplete records or missing non-neighbor parent case");
    }
    const uint64_t saved_offset = store.offsets[0];
    store.offsets[0] = store.bytes.size() + 1;
    bool rejected_bad_offset = false;
    try {
        store.decode(0, decoded.data());
    } catch (const std::runtime_error &) {
        rejected_bad_offset = true;
    }
    store.offsets[0] = saved_offset;
    if (!rejected_bad_offset) throw std::runtime_error("Out-of-bounds record was accepted");
    const uint64_t raw_bytes = uint64_t(input.count) * input.dim * sizeof(float);
    const uint64_t metadata_bytes = uint64_t(input.count) *
        (sizeof(uint32_t) + sizeof(uint8_t) + 2 * sizeof(uint64_t));
    std::cout << name << " vectors=" << input.count << '/' << input.count
              << " coordinates=" << correct_coordinates << '/' << correct_coordinates
              << " L2=" << correct_l2 << '/' << correct_l2
              << " patches=" << store.patches
              << " non_neighbor_parents=" << non_neighbor_parents
              << " payload_bytes=" << store.bytes.size()
              << " metadata_bytes=" << metadata_bytes
              << " payload_ratio=" << std::fixed << std::setprecision(4)
              << double(raw_bytes) / store.bytes.size()
              << " total_ratio=" << double(raw_bytes) / (store.bytes.size() + metadata_bytes) << '\n';
}

void test_assignment_semantics() {
    std::vector<std::vector<uint32_t>> graph(200);
    graph[0] = {1, 2, 3, 4};
    graph[100] = {101, 102, 103, 104};
    graph[1] = {0};
    graph[2] = {1};
    std::vector<float> values(200, 0);
    values[0] = 100;
    const auto neighbors = [&](uint32_t id) -> const std::vector<uint32_t> & { return graph[id]; };
    const auto vector = [&](uint32_t id) { return &values[id]; };
    const auto forest = diskann::lvc::build_forest(200, 1, neighbors, vector);
    // Node 2 reaches root 0 through node 1, although root 100 is closer.
    // Node 199 has no neighbor and therefore scans all roots.
    if (forest.parent[0] != diskann::lvc::no_parent ||
        forest.parent[100] != diskann::lvc::no_parent ||
        forest.parent[2] != 0 || forest.parent[199] != 100) {
        throw std::runtime_error("HNSW two-record assignment semantics changed");
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        test_assignment_semantics();
        const Input input = load_input(argc, argv);
        // Directed fixture: roots have degree four; child i points to i-1.
        // Node 199 has no edges and must use the global-root fallback.
        std::vector<std::vector<uint32_t>> graph(input.count);
        for (uint32_t id = 0; id < input.count; ++id) {
            if (id % 100 == 0) {
                for (uint32_t step = 1; step <= 4 && id + step < input.count; ++step) {
                    graph[id].push_back(id + step);
                }
            } else if (id != 199) {
                graph[id].push_back(id - 1);
            }
        }
        const auto vector = [&](uint32_t id) { return input.values.data() + size_t(id) * input.dim; };
        const auto neighbors = [&](uint32_t id) -> const std::vector<uint32_t> & { return graph[id]; };
        const auto forest = diskann::lvc::build_forest(input.count, input.dim, neighbors, vector);
        const uint32_t expected_roots = std::max(1U, input.count / 100);
        if (forest.initial_roots != expected_roots || forest.fallback_nodes == 0 ||
            forest.parent[199] == diskann::lvc::no_parent || forest.parent[2] != 0) {
            throw std::runtime_error("Degree roots, propagation, or fallback failed");
        }
        float best_distance = std::numeric_limits<float>::max();
        uint32_t nearest_root = diskann::lvc::no_parent;
        for (uint32_t root = 0; root < input.count; ++root) {
            if (forest.parent[root] != diskann::lvc::no_parent) continue;
            const float distance = l2(vector(199), vector(root), input.dim);
            if (distance < best_distance) {
                best_distance = distance;
                nearest_root = root;
            }
        }
        if (forest.parent[199] != nearest_root) {
            throw std::runtime_error("Fallback did not choose the nearest root");
        }
        auto invalid_forest = forest;
        invalid_forest.parent[2] = 3; // A child may only point directly to a root.
        bool rejected_bad_parent = false;
        try {
            diskann::lvc::validate_forest(invalid_forest);
        } catch (const std::runtime_error &) {
            rejected_bad_parent = true;
        }
        if (!rejected_bad_parent) throw std::runtime_error("Invalid parent was accepted");
        std::cout << "forest nodes=" << input.count << " dim=" << input.dim
                  << " initial_roots=" << forest.initial_roots
                  << " propagation_rounds=" << forest.propagation_rounds
                  << " fallback_nodes=" << forest.fallback_nodes << " max_depth=1\n";
        test_codec<hnswlib::codecs::DeXORCodecPolicy>("DeXOR", forest, input, graph);
        test_codec<hnswlib::codecs::GorillaCodecPolicy>("Gorilla", forest, input, graph);
        test_codec<hnswlib::codecs::ElfCodecPolicy>("Elf", forest, input, graph);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "D1 test failed: " << error.what() << '\n';
        return 1;
    }
}
