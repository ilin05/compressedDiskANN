// Standalone regression checks: no DiskANN/PQ/dataset dependencies.
#include "lvc_forest.h"
#include <iostream>

namespace lvc = diskann::lvc;

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
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
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
