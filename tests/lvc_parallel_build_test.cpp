// Standalone regression checks: no DiskANN/PQ/dataset dependencies.
#include "lvc_forest.h"
#include <iostream>

namespace lvc = diskann::lvc;

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Independent serial oracle: outgoing-neighbor propagation, then ascending-ID
// fallback, exact nearest root, and an ascending scan restricted to its tree.
template <typename NeighborFn, typename VectorFn>
lvc::Forest reference_root_local(const lvc::Forest &root_source, uint32_t dim,
                                  NeighborFn neighbors, VectorFn vector,
                                  uint64_t &root_distances, uint64_t &tree_distances) {
    const uint32_t n = static_cast<uint32_t>(root_source.parent.size());
    lvc::Forest result;
    result.initial_roots = root_source.initial_roots;
    result.parent.assign(n, lvc::no_parent); result.depth.assign(n, 0);
    std::vector<uint32_t> root_of(n, lvc::no_parent), roots;
    for (uint32_t i = 0; i < n; ++i) if (root_source.parent[i] == lvc::no_parent) {
        roots.push_back(i); root_of[i] = i;
    }
    auto distance = [&](uint32_t a, uint32_t b) {
        float sum = 0;
        for (uint32_t j = 0; j < dim; ++j) { const float d = vector(a)[j] - vector(b)[j]; sum += d*d; }
        return sum;
    };
    bool changed = true;
    while (changed) {
        changed = false; ++result.propagation_rounds;
        for (uint32_t i = 0; i < n; ++i) if (root_of[i] == lvc::no_parent) {
            float best = std::numeric_limits<float>::max();
            for (uint32_t candidate : neighbors(i)) if (root_of[candidate] != lvc::no_parent) {
                const float d = distance(i, candidate);
                if (d < best) { best = d; result.parent[i] = candidate; }
            }
            if (result.parent[i] != lvc::no_parent) {
                root_of[i] = root_of[result.parent[i]];
                result.depth[i] = result.depth[result.parent[i]] + 1;
                changed = true;
            }
        }
    }
    for (uint32_t i = 0; i < n; ++i) if (root_of[i] == lvc::no_parent) {
        uint32_t root = lvc::no_parent;
        float best = std::numeric_limits<float>::max();
        for (uint32_t r : roots) {
            ++root_distances;
            const float d = distance(i, r);
            if (d < best) { best = d; root = r; }
        }
        best = std::numeric_limits<float>::max();
        for (uint32_t candidate = 0; candidate < n; ++candidate) if (root_of[candidate] == root) {
            ++tree_distances;
            const float d = distance(i, candidate);
            if (d < best) { best = d; result.parent[i] = candidate; }
        }
        root_of[i] = root; result.depth[i] = result.depth[result.parent[i]] + 1;
        ++result.fallback_nodes;
    }
    lvc::validate_forest(result);
    return result;
}

// Original sequential encoding, with no root state reuse, is the byte oracle.
template <typename Codec, typename VectorFn>
lvc::RecordStore<Codec> reference_store(const lvc::Forest &forest, uint32_t dim, VectorFn vector) {
    lvc::RecordStore<Codec> store;
    store.forest = forest; store.dim = dim;
    store.offsets.resize(forest.parent.size()); store.lengths.resize(forest.parent.size());
    std::vector<uint32_t> order(forest.parent.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return forest.depth[a] != forest.depth[b] ? forest.depth[a] < forest.depth[b] : a < b;
    });
    for (uint32_t id : order) {
        std::vector<typename Codec::StateType> states(dim);
        std::vector<float> unused(dim);
        if (forest.parent[id] != lvc::no_parent) store.decode(forest.parent[id], unused.data(), states);
        auto record = lvc::encode_record<Codec>(vector(id), dim, states);
        store.offsets[id] = store.bytes.size(); store.lengths[id] = record.bytes.size();
        store.bytes.insert(store.bytes.end(), record.bytes.begin(), record.bytes.end());
        store.patches += record.patches;
    }
    return store;
}

template <typename Codec, typename VectorFn>
void check_codec(const lvc::Forest &forest, uint32_t dim, VectorFn vector, int threads) {
    const auto expected = reference_store<Codec>(forest, dim, vector);
    const auto actual = lvc::RecordStore<Codec>::build(forest, dim, vector, threads);
    require(actual.bytes == expected.bytes && actual.offsets == expected.offsets &&
            actual.lengths == expected.lengths && actual.patches == expected.patches,
            "Parallel/root-cached encoding changed bytes, offsets or patches");
    std::vector<float> decoded(dim);
    for (uint32_t i = 0; i < forest.parent.size(); ++i) {
        actual.decode(i, decoded.data());
        for (uint32_t j = 0; j < dim; ++j)
            require(lvc::bits(decoded[j]) == lvc::bits(vector(i)[j]), "Bitwise recovery failed");
    }
}

int main(int argc, char **argv) {
    const int maximum = argc > 1 ? std::stoi(argv[1]) : 32;
    try {
        lvc::validate_build_threads(maximum);
        for (int chain : {2, 4, 8, -1}) {
            // chain=2 spans multiple encoding batches; disconnected nodes force fallback.
            const uint32_t n = chain == 2 ? 4600 : 512, dim = 8;
            std::vector<std::vector<uint32_t>> graph(n);
            std::vector<float> values(size_t(n) * dim);
            for (uint32_t i = 0; i < 100; ++i) {
                if (i) graph[i].push_back(i - 1);
                if (i + 1 < 100) graph[i].push_back(i + 1);
            }
            // Repeated vectors exercise exact-distance tie handling.
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t j = 0; j < dim; ++j) values[size_t(i) * dim + j] = float((i / 2 + j * 7) % 251);
            auto neighbors = [&](uint32_t id) -> const std::vector<uint32_t>& { return graph[id]; };
            auto vector = [&](uint32_t id) { return values.data() + size_t(id) * dim; };
            const auto serial = lvc::build_forest(n, dim, neighbors, vector, chain);
            require(serial.fallback_nodes > 0, "Fixture did not exercise fallback");
            if (chain == -1) {
                uint64_t root_distances = 0, tree_distances = 0;
                const auto expected = reference_root_local(serial, dim, neighbors, vector, root_distances, tree_distances);
                for (int threads : {1, maximum}) {
                    lvc::ForestBuildTimings timings;
                    const auto local = lvc::build_forest(n, dim, neighbors, vector, chain, &timings, threads, 1);
                    require(local.parent == expected.parent && local.depth == expected.depth &&
                            local.fallback_nodes == expected.fallback_nodes &&
                            local.propagation_rounds == expected.propagation_rounds,
                            "Root-local fallback differs from serial specification");
                    require(timings.fallback_root_distances == root_distances &&
                            timings.fallback_tree_distances == tree_distances, "Incorrect candidate counters");
                    check_codec<hnswlib::codecs::DeXORCodecPolicy>(local, dim, vector, threads);
                    check_codec<hnswlib::codecs::GorillaCodecPolicy>(local, dim, vector, threads);
                    check_codec<hnswlib::codecs::ElfCodecPolicy>(local, dim, vector, threads);
                }
                std::cout << "PASS nearest-root fallback: serial oracle, 1/" << maximum
                          << " threads, distance counts, lossless codec bytes\n";
            }
            for (int threads : {1, maximum}) {
                const auto parallel = lvc::build_forest(n, dim, neighbors, vector, chain, nullptr, threads);
                require(parallel.parent == serial.parent && parallel.depth == serial.depth &&
                        parallel.propagation_rounds == serial.propagation_rounds &&
                        parallel.fallback_nodes == serial.fallback_nodes, "Parallel fallback changed the forest");
                check_codec<hnswlib::codecs::DeXORCodecPolicy>(parallel, dim, vector, threads);
                check_codec<hnswlib::codecs::GorillaCodecPolicy>(parallel, dim, vector, threads);
                check_codec<hnswlib::codecs::ElfCodecPolicy>(parallel, dim, vector, threads);
            }
            // Exceptions inside worker loops must reach the caller safely.
            bool caught = false;
            auto bad_vector = [&](uint32_t) -> const float* { throw std::runtime_error("worker error"); };
            try {
                lvc::RecordStore<hnswlib::codecs::DeXORCodecPolicy>::build(serial, dim, bad_vector, maximum);
            } catch (const std::runtime_error &) { caught = true; }
            require(caught, "Worker exception was not propagated");
            caught = false;
            auto bad_fallback = [&](uint32_t id) -> const float* {
                if (id == n - 1) throw std::runtime_error("fallback worker error");
                return vector(id);
            };
            try {
                lvc::build_forest(n, dim, neighbors, bad_fallback, chain, nullptr, maximum);
            } catch (const std::runtime_error &) { caught = true; }
            require(caught, "Fallback worker exception was not propagated");
            std::cout << "PASS chain=" << chain << " threads=1/" << maximum
                      << " deterministic forest, legacy codec bytes, bitwise recovery, worker exceptions\n";
        }
        {
            // A globally nearest predecessor can belong to a different root.
            std::vector<std::vector<uint32_t>> empty(200);
            std::vector<float> values(200, 1000);
            values[0] = 0; values[1] = 10; values[2] = 6; values[3] = 4.9f; values[4] = 5;
            auto neighbors = [&](uint32_t id) -> const std::vector<uint32_t>& { return empty[id]; };
            auto vector = [&](uint32_t id) { return &values[id]; };
            const auto global = lvc::build_forest(200, 1, neighbors, vector, -1);
            const auto local = lvc::build_forest(200, 1, neighbors, vector, -1, nullptr, maximum, 1);
            require(global.parent[3] == 2 && local.parent[3] == 0,
                    "Nearest-root restriction is not enforced");
            require(local.parent[4] == 3, "Equal root distances must select root ID 0");
            std::cout << "PASS root-local restriction and root-distance ties\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
