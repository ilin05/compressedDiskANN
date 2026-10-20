#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "tsl/robin_map.h"

namespace diskann {
using CacheNhoodLookup = tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>>;
using CacheCoordLookup = tsl::robin_map<uint32_t, size_t>;

// DiskANN bin format: uint32 count, uint32 dim=1, followed by sorted uint32 IDs.
inline std::vector<uint32_t> read_cache_ids(const std::string &path, uint64_t take,
                                           uint64_t points = UINT64_MAX) {
    std::ifstream in(path, std::ios::binary);
    uint32_t count = 0, dim = 0;
    if (!in.read(reinterpret_cast<char *>(&count), 4) ||
        !in.read(reinterpret_cast<char *>(&dim), 4) || dim != 1 ||
        std::filesystem::file_size(path) != 8ULL + uint64_t(count) * 4 || take > count)
        throw std::runtime_error("Invalid/insufficient cache ID file: " + path);
    std::vector<uint32_t> ids(take);
    if (take && !in.read(reinterpret_cast<char *>(ids.data()), take * 4))
        throw std::runtime_error("Truncated cache IDs");
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] >= points || (i && ids[i] <= ids[i - 1]))
            throw std::runtime_error("Cache IDs must be unique, ascending and in range");
    return ids;
}

inline void write_cache_ids(const std::string &path, const std::vector<uint32_t> &ids) {
    if (ids.size() > UINT32_MAX) throw std::runtime_error("Too many cache IDs");
    const std::string temporary = path + ".pending";
    std::ofstream out(temporary, std::ios::binary);
    uint32_t count = static_cast<uint32_t>(ids.size()), dim = 1;
    out.write(reinterpret_cast<const char *>(&count), 4);
    out.write(reinterpret_cast<const char *>(&dim), 4);
    out.write(reinterpret_cast<const char *>(ids.data()), ids.size() * 4);
    if (!out) throw std::runtime_error("Cannot save cache IDs: " + path);
    out.close();
    if (!out) throw std::runtime_error("Cannot finish cache IDs: " + path);
    std::filesystem::rename(temporary, path);
}

inline uint64_t cache_lookup_bytes(const CacheNhoodLookup &nhood, const CacheCoordLookup &coord) {
    return nhood.bucket_count() * sizeof(tsl::detail_robin_hash::bucket_entry<CacheNhoodLookup::value_type, false>) +
           coord.bucket_count() * sizeof(tsl::detail_robin_hash::bucket_entry<CacheCoordLookup::value_type, false>);
}
} // namespace diskann
