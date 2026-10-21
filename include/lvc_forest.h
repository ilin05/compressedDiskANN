#pragma once

#include <algorithm>
#include <chrono>
#include <exception>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "lvc_f32_codec.h"

namespace diskann::lvc {

constexpr uint32_t no_parent = UINT32_MAX;

inline void validate_build_threads(int threads) {
    if (threads < 1) throw std::invalid_argument("LVC build threads must be positive");
#ifndef _OPENMP
    if (threads != 1) throw std::invalid_argument("Parallel LVC build requires OpenMP");
#endif
}

// Coarse wall-clock timers only: no timing calls in node/distance loops.
struct ForestBuildTimings {
    double root_selection_seconds = 0; // Includes allocation and root initialization.
    double propagation_seconds = 0;
    double fallback_seconds = 0;
    double finalize_seconds = 0; // Parent materialization and validation.
    double fallback_bucket_seconds = 0;
    double fallback_root_search_seconds = 0;
    double fallback_tree_search_seconds = 0;
    uint64_t fallback_root_distances = 0, fallback_tree_distances = 0;
    uint64_t fallback_max_tree_nodes = 0;
};

struct Forest {
    std::vector<uint32_t> parent; // Local node IDs; no_parent marks a root.
    std::vector<uint32_t> depth;  // Number of edges from a root.
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

    // Vector allocations plus conservative unordered-map nodes/buckets.
    // Allocator headers are not included; process RSS is reported separately.
    size_t resident_bytes() const {
        size_t bytes = entries.bucket_count() * sizeof(void*);
        for (const auto &item : entries)
            bytes += item.second.states.capacity() * sizeof(typename Codec::StateType) +
                     item.second.vector.capacity() * sizeof(float) +
                     sizeof(typename decltype(entries)::value_type) + 2 * sizeof(void*);
        return entries.empty() ? 0 : bytes;
    }

  private:
    std::unordered_map<uint32_t, Entry> entries;
};

inline void validate_forest(const Forest &forest) {
    // Every parent must have depth exactly one less than its child; a cycle
    // would require depth to decrease forever, so this also checks acyclicity.
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
        } else if (forest.parent[i] >= count || forest.parent[i] == i || forest.depth[i] == 0 ||
                   forest.depth[i] >= count ||
                   uint64_t(forest.depth[forest.parent[i]]) + 1 != forest.depth[i]) {
            throw std::runtime_error("Invalid LVC parent or depth");
        }
    }
    if (roots != forest.initial_roots) throw std::runtime_error("LVC root count changed");
}

// Root-local fallback for unbounded chains. Only membership in the predecessor
// tree matters here, not outgoing reachability in the original graph.
template <typename DistanceFn>
void assign_root_local_fallback(Forest &forest, std::vector<uint32_t> &root_of,
                               const std::vector<uint8_t> &is_root,
                               const std::vector<uint32_t> &unresolved, DistanceFn distance,
                               int build_threads, ForestBuildTimings *timings) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    std::vector<uint32_t> roots, slot(root_of.size(), no_parent);
    for (uint32_t i = 0; i < root_of.size(); ++i) {
        if (is_root[i]) { slot[i] = static_cast<uint32_t>(roots.size()); roots.push_back(i); }
    }
    std::vector<std::vector<uint32_t>> members(roots.size()), pending(roots.size());
    for (uint32_t i = 0; i < root_of.size(); ++i)
        if (root_of[i] != no_parent) members[slot[root_of[i]]].push_back(i);
    std::vector<uint32_t> nearest_slot(unresolved.size(), no_parent);
    auto now = Clock::now();
    double bucket_seconds = std::chrono::duration<double>(now - start).count();
    start = now;
    std::exception_ptr failure;
#pragma omp parallel for num_threads(build_threads) schedule(static) if(build_threads > 1)
    for (size_t index = 0; index < unresolved.size(); ++index) {
        try {
            float best = std::numeric_limits<float>::max();
            for (uint32_t j = 0; j < roots.size(); ++j) {
                const float d = distance(unresolved[index], roots[j]);
                // roots are ascending: exact ties select the smallest root ID.
                if (d < best) { best = d; nearest_slot[index] = j; }
            }
            if (nearest_slot[index] == no_parent) throw std::runtime_error("No finite LVC root distance");
        } catch (...) {
#pragma omp critical(lvc_build_failure)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
    now = Clock::now();
    if (timings) {
        timings->fallback_root_search_seconds = std::chrono::duration<double>(now - start).count();
        timings->fallback_root_distances = uint64_t(unresolved.size()) * roots.size();
    }
    start = now;
    // unresolved is ascending. Preserve that order within every root bucket.
    for (size_t i = 0; i < unresolved.size(); ++i) pending[nearest_slot[i]].push_back(unresolved[i]);
    uint64_t tree_distances = 0, max_tree_nodes = 0;
    for (size_t j = 0; j < roots.size(); ++j) {
        const uint64_t f = pending[j].size(), n = members[j].size();
        tree_distances += f * n + (f ? f * (f - 1) / 2 : 0);
        max_tree_nodes = std::max(max_tree_nodes, n + f);
        members[j].reserve(n + f);
    }
    now = Clock::now();
    bucket_seconds += std::chrono::duration<double>(now - start).count();
    start = now;
    // Different root trees are independent. Within a tree, only nodes already
    // assigned (including earlier fallback nodes) can become a predecessor.
#pragma omp parallel for num_threads(build_threads) schedule(dynamic, 1) if(build_threads > 1)
    for (size_t j = 0; j < roots.size(); ++j) {
        try {
            auto &candidates = members[j];
            for (uint32_t id : pending[j]) {
                float best = std::numeric_limits<float>::max();
                uint32_t parent = no_parent;
                for (uint32_t candidate : candidates) {
                    const float d = distance(id, candidate);
                    if (d < best || (parent != no_parent && d == best && candidate < parent)) {
                        best = d; parent = candidate;
                    }
                }
                if (parent == no_parent) throw std::runtime_error("No finite LVC predecessor distance");
                forest.parent[id] = parent;
                forest.depth[id] = forest.depth[parent] + 1;
                root_of[id] = roots[j];
                candidates.push_back(id);
            }
        } catch (...) {
#pragma omp critical(lvc_build_failure)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
    forest.fallback_nodes = static_cast<uint32_t>(unresolved.size());
    if (timings) {
        timings->fallback_bucket_seconds = bucket_seconds;
        timings->fallback_tree_search_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        timings->fallback_tree_distances = tree_distances;
        timings->fallback_max_tree_nodes = max_tree_nodes;
    }
}

// Neighbors and vectors use dense local IDs. For a disk cache, callers must
// filter neighbors to cached nodes and map global IDs to local IDs first.
template <typename NeighborFn, typename VectorFn>
Forest build_forest(uint32_t count, uint32_t dim, NeighborFn neighbors, VectorFn vector,
                    int chain_max = 2, ForestBuildTimings *timings = nullptr, int build_threads = 1,
                    int fallback_root_candidates = 0) {
    validate_build_threads(build_threads);
    if (fallback_root_candidates != 0 && fallback_root_candidates != 1)
        throw std::invalid_argument("LVC fallback root candidates must be 0 (global) or 1 (nearest root)");
    if (count == 0 || dim == 0) throw std::runtime_error("Empty LVC forest input");
    if (chain_max != -1 && chain_max < 2) throw std::runtime_error("LVC chain_max must be -1 or >= 2");
    using Clock = std::chrono::steady_clock;
    auto stage_start = Clock::now();
    auto finish_stage = [&](double ForestBuildTimings::*field) {
        const auto now = Clock::now();
        if (timings) timings->*field = std::chrono::duration<double>(now - stage_start).count();
        stage_start = now;
    };
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
    finish_stage(&ForestBuildTimings::root_selection_seconds);

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

    if (chain_max != 2) {
        // Match HNSW's deeper-chain policy: assign each unvisited node to its
        // nearest already assigned graph neighbor, respecting the depth limit.
        std::vector<uint8_t> visited = is_root;
        bool changed = true;
        while (changed) {
            changed = false;
            ++forest.propagation_rounds;
            for (uint32_t i = 0; i < count; ++i) {
                if (visited[i]) continue;
                uint32_t best_parent = no_parent;
                float best = std::numeric_limits<float>::max();
                for (uint32_t neighbor : neighbors(i)) {
                    if (neighbor >= count) throw std::runtime_error("LVC neighbor ID out of range");
                    if (!visited[neighbor] || (chain_max > 0 && forest.depth[neighbor] >= uint32_t(chain_max - 1))) continue;
                    const float d = distance(i, neighbor);
                    if (d < best) { best = d; best_parent = neighbor; }
                }
                if (best_parent != no_parent) {
                    forest.parent[i] = best_parent;
                    forest.depth[i] = forest.depth[best_parent] + 1;
                    assigned[i] = assigned[best_parent];
                    visited[i] = 1;
                    changed = true;
                }
            }
        }
        finish_stage(&ForestBuildTimings::propagation_seconds);
        // Match HNSW's fallback: scan already assigned nodes with room in the
        // chain. A root is always eligible, so root count remains fixed.
        std::vector<uint32_t> unresolved;
        for (uint32_t i = 0; i < count; ++i) if (!visited[i]) unresolved.push_back(i);
        if (chain_max == -1 && fallback_root_candidates == 1) {
            assign_root_local_fallback(forest, assigned, is_root, unresolved, distance, build_threads, timings);
            finish_stage(&ForestBuildTimings::fallback_seconds);
            validate_forest(forest);
            finish_stage(&ForestBuildTimings::finalize_seconds);
            return forest;
        }
        std::vector<float> best_distances(build_threads);
        std::vector<uint32_t> best_parents(build_threads);
        std::exception_ptr failure;
        bool invalid_distance = false;
        // Keep assignment order serial: a newly assigned fallback node is an
        // eligible predecessor for the next node. Reuse one OpenMP team.
#pragma omp parallel num_threads(build_threads) if(build_threads > 1 && !unresolved.empty())
        {
            int worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            for (uint32_t i : unresolved) {
                float best = std::numeric_limits<float>::max();
                uint32_t parent = no_parent;
#pragma omp for schedule(static) nowait
                for (uint32_t candidate = 0; candidate < count; ++candidate) {
                    if (!visited[candidate] ||
                        (chain_max > 0 && forest.depth[candidate] >= uint32_t(chain_max - 1))) continue;
                    try {
                        const float d = distance(i, candidate);
                        if (d < best) { best = d; parent = candidate; }
                    } catch (...) {
#pragma omp critical(lvc_build_failure)
                        { if (!failure) failure = std::current_exception(); }
                    }
                }
                best_distances[worker] = best;
                best_parents[worker] = parent;
#pragma omp barrier
#pragma omp single
                {
                    float value = std::numeric_limits<float>::max();
                    uint32_t selected = no_parent;
                    int workers = 1;
#ifdef _OPENMP
                    workers = omp_get_num_threads();
#endif
                    for (int thread = 0; thread < workers; ++thread) {
                        const float d = best_distances[thread];
                        const uint32_t p = best_parents[thread];
                        if (p != no_parent && (d < value || (d == value && p < selected))) {
                            value = d; selected = p;
                        }
                    }
                    if (selected == no_parent) invalid_distance = true;
                    else {
                        forest.parent[i] = selected;
                        forest.depth[i] = forest.depth[selected] + 1;
                        visited[i] = 1;
                        ++forest.fallback_nodes;
                    }
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
        if (invalid_distance) throw std::runtime_error("No finite LVC root distance");
        finish_stage(&ForestBuildTimings::fallback_seconds);
        validate_forest(forest);
        finish_stage(&ForestBuildTimings::finalize_seconds);
        return forest;
    }

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

    finish_stage(&ForestBuildTimings::propagation_seconds);
    // Only nodes unreached by propagation scan all initial roots.
    std::vector<uint32_t> root_ids, unresolved;
    for (uint32_t i = 0; i < count; ++i) {
        if (is_root[i]) root_ids.push_back(i); // Ascending ID preserves distance ties.
        else if (assigned[i] == no_parent) unresolved.push_back(i);
    }
    std::exception_ptr failure;
    int invalid_distance = 0;
#pragma omp parallel for num_threads(build_threads) schedule(static) if(build_threads > 1) reduction(|:invalid_distance)
    for (size_t index = 0; index < unresolved.size(); ++index) {
        const uint32_t i = unresolved[index];
        try {
            float best = std::numeric_limits<float>::max();
            for (uint32_t root : root_ids) {
                const float d = distance(i, root);
                if (d < best) { best = d; assigned[i] = root; }
            }
            if (assigned[i] == no_parent) invalid_distance = 1;
        } catch (...) {
#pragma omp critical(lvc_build_failure)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
    if (invalid_distance) throw std::runtime_error("No finite LVC root distance");
    forest.fallback_nodes = static_cast<uint32_t>(unresolved.size());
    finish_stage(&ForestBuildTimings::fallback_seconds);

    for (uint32_t i = 0; i < count; ++i) {
        if (!is_root[i]) {
            forest.parent[i] = assigned[i];
            forest.depth[i] = 1;
        }
    }
    validate_forest(forest);
    finish_stage(&ForestBuildTimings::finalize_seconds);
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
    static RecordStore build(const Forest &source, uint32_t dimension, VectorFn vector,
                             int build_threads = 1) {
        validate_build_threads(build_threads);
        validate_forest(source);
        if (dimension == 0) throw std::runtime_error("Zero LVC dimension");
        RecordStore store;
        store.forest = source;
        store.dim = dimension;
        store.offsets.resize(source.parent.size());
        store.lengths.resize(source.parent.size());

        std::vector<uint32_t> order(source.parent.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return source.depth[a] != source.depth[b] ? source.depth[a] < source.depth[b] : a < b;
        });
        RootStateCache<Codec> root_states;
        root_states.reserve(source.initial_roots);
        // Bounded temporary records, independent of the total node count.
        constexpr size_t batch_size = 4096;
        for (size_t level_begin = 0; level_begin < order.size();) {
            size_t level_end = level_begin + 1;
            while (level_end < order.size() && source.depth[order[level_end]] == source.depth[order[level_begin]])
                ++level_end;
            for (size_t begin = level_begin; begin < level_end; begin += batch_size) {
                const size_t size = std::min(batch_size, level_end - begin);
                std::vector<EncodedRecord> records(size);
                std::exception_ptr failure;
                // Shared payload is immutable throughout this region. Parent
                // records are in earlier levels and the root cache is read-only.
#pragma omp parallel for num_threads(build_threads) schedule(static) if(build_threads > 1)
                for (size_t j = 0; j < size; ++j) {
                    try {
                        const uint32_t id = order[begin + j];
                        std::vector<typename Codec::StateType> states(dimension);
                        if (source.parent[id] != no_parent) {
                            std::vector<float> unused(dimension);
                            store.decode(source.parent[id], unused.data(), states, &root_states);
                        }
                        records[j] = encode_record<Codec>(vector(id), dimension, states);
                    } catch (...) {
#pragma omp critical(lvc_build_failure)
                        { if (!failure) failure = std::current_exception(); }
                    }
                }
                if (failure) std::rethrow_exception(failure);
                size_t end = store.bytes.size();
                for (size_t j = 0; j < size; ++j) {
                    const uint32_t id = order[begin + j];
                    store.offsets[id] = end;
                    store.lengths[id] = records[j].bytes.size();
                    end += records[j].bytes.size();
                    store.patches += records[j].patches;
                }
                // No decoder is active while resize can relocate the payload.
                store.bytes.resize(end);
#pragma omp parallel for num_threads(build_threads) schedule(static) if(build_threads > 1)
                for (size_t j = 0; j < size; ++j) {
                    const uint32_t id = order[begin + j];
                    std::copy(records[j].bytes.begin(), records[j].bytes.end(),
                              store.bytes.begin() + store.offsets[id]);
                }
            }
            if (source.depth[order[level_begin]] == 0) {
                // Canonical decoded states, not encoder states or F32 patches.
                for (size_t j = level_begin; j < level_end; ++j) {
                    const uint32_t id = order[j];
                    root_states.insert(id, store.bytes.data() + store.offsets[id], store.lengths[id], dimension);
                }
            }
            level_begin = level_end;
        }
        return store;
    }

    size_t metadata_bytes() const {
        return (offsets.capacity() + lengths.capacity()) * sizeof(uint64_t) +
               (forest.parent.capacity() + forest.depth.capacity()) * sizeof(uint32_t);
    }

    void decode(uint32_t id, float *output) const {
        std::vector<typename Codec::StateType> states(dim);
        decode(id, output, states);
    }

    void decode(uint32_t id, float *output, std::vector<typename Codec::StateType> &states,
                const RootStateCache<Codec> *cache = nullptr, uint64_t *records = nullptr,
                uint64_t *bytes_read = nullptr, bool *state_hit = nullptr) const {
        if (id >= forest.parent.size() || output == nullptr) throw std::runtime_error("Invalid LVC decode target");
        std::vector<uint32_t> path;
        for (uint32_t current = id; current != no_parent; current = forest.parent[current])
            path.push_back(current);
        const uint32_t root = path.back();
        const auto *entry = cache == nullptr ? nullptr : cache->find(root);
        if (state_hit != nullptr) *state_hit = entry != nullptr;
        if (bytes_read != nullptr) *bytes_read = 0;
        if (path.size() == 1 && entry != nullptr) {
            states = entry->states;
            std::copy(entry->vector.begin(), entry->vector.end(), output);
            if (records != nullptr) *records = 0;
            return;
        }
        states = entry != nullptr ? entry->states : std::vector<typename Codec::StateType>(dim);
        uint64_t replayed = 0;
        for (auto it = path.rbegin() + (entry != nullptr ? 1 : 0); it != path.rend(); ++it) {
            decode_one(*it, states, output);
            if (bytes_read != nullptr) *bytes_read += lengths[*it];
            ++replayed;
        }
        if (records != nullptr) *records = replayed;
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
