#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include "cache_lookup_layout.h"

namespace diskann {
// No reserve/rehash: match a fresh load_cache_list's map types and coord-first insertion.
inline uint64_t probe_cache_lookup(const std::vector<uint32_t> &ids, uint64_t limit,
                                  const std::string &csv_path, bool verify) {
    CacheNhoodLookup nhood;
    CacheCoordLookup coord;
    std::ofstream csv(csv_path);
    if (!csv || !limit) throw std::runtime_error("Invalid probe CSV or byte limit");
    csv << "first_nodes,last_nodes,nhood_buckets,coord_buckets,lookup_bytes,"
           "nhood_bucket_bytes,coord_bucket_bytes,prefix_hash,reason,elapsed_seconds\n";
    uint64_t first = 0, old_n = nhood.bucket_count(), old_c = coord.bucket_count();
    uint64_t old_bytes = cache_lookup_bytes(nhood, coord), hash = 1469598103934665603ULL;
    uint64_t checked = 0;
    const auto start = std::chrono::steady_clock::now();
    struct Check { uint64_t n, nb, cb; };
    std::vector<Check> checks;
    auto emit = [&](uint64_t last, uint64_t h, const char *reason) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        csv << first << ',' << last << ',' << old_n << ',' << old_c << ',' << old_bytes << ','
            << sizeof(tsl::detail_robin_hash::bucket_entry<CacheNhoodLookup::value_type, false>) << ','
            << sizeof(tsl::detail_robin_hash::bucket_entry<CacheCoordLookup::value_type, false>) << ','
            << h << ',' << reason << ',' << elapsed << '\n';
        csv.flush();
        if (!csv) throw std::runtime_error("Probe CSV write failed");
        if (verify) checks.push_back({last, old_n, old_c});
    };
    bool exceeded = false;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i && ids[i] <= ids[i - 1]) throw std::runtime_error("Probe requires sorted unique IDs");
        const uint64_t before_hash = hash, n = i + 1;
        // Values do not affect uint32-key hashing; no payload/neighbor allocation is needed.
        coord.insert(std::make_pair(ids[i], i));
        nhood.insert(std::make_pair(ids[i], std::make_pair(0U, static_cast<uint32_t *>(nullptr))));
        hash = (hash ^ ids[i]) * 1099511628211ULL;
        const uint64_t nb = nhood.bucket_count(), cb = coord.bucket_count();
        if (nb != old_n || cb != old_c) {
            emit(n - 1, before_hash, "before_growth");
            first = n; old_n = nb; old_c = cb;
            old_bytes = cache_lookup_bytes(nhood, coord);
            std::cout << "LOOKUP_GROW nodes=" << n << " bytes=" << old_bytes
                      << " nhood_buckets=" << nb << " coord_buckets=" << cb << std::endl;
        }
        checked = n;
        if (old_bytes > limit) {
            emit(n, hash, "limit_exceeded");
            exceeded = true;
            break;
        }
        if (n % 1000000 == 0)
            std::cout << "LOOKUP_PROGRESS nodes=" << n << " bytes=" << old_bytes << std::endl;
    }
    if (!exceeded) emit(checked, hash, "end_of_ids");
    if (verify) {
        // Independent fresh-map replay validates the interval endpoints and ID/value retrieval.
        // Small correctness runs only: do not double the 1B experiment's map footprint.
        nhood = CacheNhoodLookup(); coord = CacheCoordLookup();
        size_t position = 0;
        for (const auto &check : checks) {
            while (position < check.n) {
                const uint32_t id = ids[position];
                coord.insert(std::make_pair(id, position));
                nhood.insert(std::make_pair(id, std::make_pair(0U, static_cast<uint32_t *>(nullptr))));
                if (coord.at(id) != position || nhood.at(id).first != 0)
                    throw std::runtime_error("Lookup replay retrieval mismatch");
                ++position;
            }
            if (nhood.bucket_count() != check.nb || coord.bucket_count() != check.cb)
                throw std::runtime_error("Lookup interval replay mismatch");
        }
        std::cout << "PASS lookup replay endpoints=" << checks.size() << " nodes=" << checked << std::endl;
    }
    std::cout << (exceeded ? "LOOKUP_LIMIT_EXCEEDED" : "LOOKUP_IDS_EXHAUSTED")
              << " nodes=" << checked << " bytes=" << old_bytes << " limit=" << limit << std::endl;
    return checked;
}
} // namespace diskann
