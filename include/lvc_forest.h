#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lvc_f32_codec.h"

namespace diskann::lvc {

constexpr uint32_t no_parent = UINT32_MAX;

struct Forest {
    std::vector<uint32_t> parent; // Local node IDs; no_parent marks a root.
    std::vector<uint8_t> depth;   // 0 for roots, 1 for children.
    uint32_t initial_roots = 0;
    uint32_t fallback_nodes = 0;
    uint32_t propagation_rounds = 0;
};

template <typename Codec> class RootStateCache {
  public:
    struct Entry {
        std::vector<typename Codec::StateType> states;
        std::vector<float> vector;
    };

    void reserve(size_t count) { entries.reserve(count); }

    void insert(uint32_t id, const uint8_t *record, size_t length, uint32_t dim) {
        Entry entry;
        entry.states.resize(dim);
        entry.vector.resize(dim);
        decode_record<Codec>(record, length, entry.states, entry.vector.data());
        entries.emplace(id, std::move(entry));
    }

    const Entry *find(uint32_t id) const {
        const auto it = entries.find(id);
        return it == entries.end() ? nullptr : &it->second;
    }

    size_t size() const { return entries.size(); }

    size_t payload_bytes() const {
        if (entries.empty()) return 0;
        return entries.size() * (entries.begin()->second.states.size() * sizeof(typename Codec::StateType) +
                                 entries.begin()->second.vector.size() * sizeof(float));
    }

  private:
    std::unordered_map<uint32_t, Entry> entries;
};

inline void validate_forest(const Forest &forest) {
    const size_t count = forest.parent.size();
    if (count == 0 || forest.depth.size() != count || forest.initial_roots == 0 ||
        forest.initial_roots > count) {
        throw std::runtime_error("Invalid LVC forest size or root count");
    }
    size_t roots = 0;
    for (size_t i = 0; i < count; ++i) {
        if (forest.parent[i] == no_parent) {
            if (forest.depth[i] != 0) throw std::runtime_error("Invalid root depth");
            ++roots;
        } else if (forest.parent[i] >= count || forest.parent[i] == i || forest.depth[i] != 1 ||
                   forest.parent[forest.parent[i]] != no_parent) {
            throw std::runtime_error("Invalid LVC parent or depth");
        }
    }
    if (roots != forest.initial_roots) throw std::runtime_error("LVC root count changed");
}

// Neighbors and vectors use dense local IDs. For a disk cache, callers must
// filter neighbors to cached nodes and map global IDs to local IDs first.
template <typename NeighborFn, typename VectorFn>
Forest build_forest(uint32_t count, uint32_t dim, NeighborFn neighbors, VectorFn vector) {
    if (count == 0 || dim == 0) throw std::runtime_error("Empty LVC forest input");
    Forest forest;
    forest.parent.assign(count, no_parent);
    forest.depth.assign(count, 0);

    std::vector<uint32_t> rank(count);
    std::iota(rank.begin(), rank.end(), 0);
    std::sort(rank.begin(), rank.end(), [&](uint32_t a, uint32_t b) {
        const auto da = neighbors(a).size();
        const auto db = neighbors(b).size();
        return da != db ? da > db : a < b;
    });
    // Match HNSW's floor(N * 0.01), with at least one root.
    forest.initial_roots = std::max<uint32_t>(1, count / 100);
    std::vector<uint8_t> is_root(count, 0);
    std::vector<uint32_t> assigned(count, no_parent);
    std::vector<float> min_dist(count, std::numeric_limits<float>::max());
    for (uint32_t i = 0; i < forest.initial_roots; ++i) {
        const uint32_t root = rank[i];
        is_root[root] = 1;
        assigned[root] = root;
        min_dist[root] = 0;
    }

    auto distance = [&](uint32_t a, uint32_t b) {
        const float *va = vector(a);
        const float *vb = vector(b);
        float sum = 0;
        for (uint32_t j = 0; j < dim; ++j) {
            const float diff = va[j] - vb[j];
            sum += diff * diff;
        }
        return sum;
    };

    // HNSW's chain_max_length == 2 propagation: compare the roots already
    // assigned to immediate neighbors, not the neighbors themselves.
    bool changed = true;
    for (uint32_t iter = 0; iter < 100 && changed; ++iter) {
        changed = false;
        ++forest.propagation_rounds;
        for (uint32_t i = 0; i < count; ++i) {
            if (is_root[i]) continue;
            for (uint32_t neighbor : neighbors(i)) {
                if (neighbor >= count) throw std::runtime_error("LVC neighbor ID out of range");
                const uint32_t root = assigned[neighbor];
                if (root == no_parent || root == assigned[i]) continue;
                const float d = distance(i, root);
                if (d < min_dist[i]) {
                    min_dist[i] = d;
                    assigned[i] = root;
                    changed = true;
                }
            }
        }
    }

    // Only nodes unreached by propagation scan all initial roots.
    for (uint32_t i = 0; i < count; ++i) {
        if (is_root[i] || assigned[i] != no_parent) continue;
        float best = std::numeric_limits<float>::max();
        for (uint32_t root = 0; root < count; ++root) {
            if (!is_root[root]) continue;
            const float d = distance(i, root);
            if (d < best) {
                best = d;
                assigned[i] = root;
            }
        }
        if (assigned[i] == no_parent) throw std::runtime_error("No finite LVC root distance");
        ++forest.fallback_nodes;
    }

    for (uint32_t i = 0; i < count; ++i) {
        if (!is_root[i]) {
            forest.parent[i] = assigned[i];
            forest.depth[i] = 1;
        }
    }
    validate_forest(forest);
    return forest;
}

template <typename Codec> class RecordStore {
  public:
    Forest forest;
    uint32_t dim = 0;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> lengths;
    std::vector<uint8_t> bytes;
    uint64_t patches = 0;

    template <typename VectorFn>
    static RecordStore build(const Forest &source, uint32_t dimension, VectorFn vector) {
        validate_forest(source);
        if (dimension == 0) throw std::runtime_error("Zero LVC dimension");
        RecordStore store;
        store.forest = source;
        store.dim = dimension;
        store.offsets.resize(source.parent.size());
        store.lengths.resize(source.parent.size());

        auto append = [&](uint32_t id, const std::vector<typename Codec::StateType> &states) {
            auto record = encode_record<Codec>(vector(id), dimension, states);
            store.offsets[id] = store.bytes.size();
            store.lengths[id] = record.bytes.size();
            store.bytes.insert(store.bytes.end(), record.bytes.begin(), record.bytes.end());
            store.patches += record.patches;
        };

        for (uint32_t id = 0; id < source.parent.size(); ++id) {
            if (source.parent[id] == no_parent) {
                append(id, std::vector<typename Codec::StateType>(dimension));
            }
        }
        for (uint32_t id = 0; id < source.parent.size(); ++id) {
            const uint32_t root = source.parent[id];
            if (root == no_parent) continue;
            std::vector<typename Codec::StateType> states(dimension);
            store.decode_one(root, states, nullptr);
            append(id, states);
        }
        return store;
    }

    void decode(uint32_t id, float *output) const {
        std::vector<typename Codec::StateType> states(dim);
        decode(id, output, states);
    }

    void decode(uint32_t id, float *output, std::vector<typename Codec::StateType> &states,
                const RootStateCache<Codec> *cache = nullptr, uint64_t *records = nullptr) const {
        if (id >= forest.parent.size() || output == nullptr) throw std::runtime_error("Invalid LVC decode target");
        const uint32_t root = forest.parent[id];
        const auto *entry = cache == nullptr ? nullptr : cache->find(root == no_parent ? id : root);
        if (root == no_parent && entry != nullptr) {
            states = entry->states;
            std::copy(entry->vector.begin(), entry->vector.end(), output);
            if (records != nullptr) *records = 0;
            return;
        }
        if (entry != nullptr) states = entry->states;
        else {
            states.assign(dim, typename Codec::StateType{});
            if (root != no_parent) decode_one(root, states, nullptr);
        }
        decode_one(id, states, output);
        if (records != nullptr) *records = 1 + (root != no_parent && entry == nullptr ? 1 : 0);
    }

  private:
    void decode_one(uint32_t id, std::vector<typename Codec::StateType> &states, float *output) const {
        if (id >= offsets.size() || lengths.size() != offsets.size() ||
            offsets[id] > bytes.size() || lengths[id] > bytes.size() - offsets[id]) {
            throw std::runtime_error("LVC record offset out of bounds");
        }
        decode_record<Codec>(bytes.data() + offsets[id], lengths[id], states, output);
    }
};

} // namespace diskann::lvc
