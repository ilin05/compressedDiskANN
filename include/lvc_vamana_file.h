#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lvc_forest.h"

namespace diskann::lvc {

constexpr uint64_t vamana_magic = 0x314154414456434cULL;
constexpr uint32_t vamana_version = 1;

inline void write_header(std::ofstream &out, uint32_t count, uint32_t dim, uint32_t mode,
                         const std::vector<uint32_t> &parent, const std::vector<uint8_t> &depth) {
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char *>(&vamana_magic), sizeof(vamana_magic));
    out.write(reinterpret_cast<const char *>(&vamana_version), sizeof(vamana_version));
    out.write(reinterpret_cast<const char *>(&mode), sizeof(mode));
    if (mode != 1) {
        out.write(reinterpret_cast<const char *>(parent.data()), count * sizeof(uint32_t));
        out.write(reinterpret_cast<const char *>(depth.data()), count);
    }
}

inline void write_raw_file(const std::string &path, uint32_t count, uint32_t dim,
                           const std::vector<float> &vectors) {
    if (vectors.size() != uint64_t(count) * dim) throw std::runtime_error("Raw LVC vector count mismatch");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create " + path);
    const std::vector<uint32_t> parent;
    const std::vector<uint8_t> depth;
    write_header(out, count, dim, 1, parent, depth);
    const uint64_t size = vectors.size() * sizeof(float);
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    out.write(reinterpret_cast<const char *>(vectors.data()), size);
    if (!out) throw std::runtime_error("Cannot write " + path);
}

template <typename Codec>
void write_record_file(const std::string &path, uint32_t count, uint32_t dim, uint32_t mode,
                       const RecordStore<Codec> &store) {
    if (store.forest.parent.size() != count || store.dim != dim || store.offsets.size() != count ||
        store.lengths.size() != count) throw std::runtime_error("LVC record count mismatch");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create " + path);
    write_header(out, count, dim, mode, store.forest.parent, store.forest.depth);
    out.write(reinterpret_cast<const char *>(store.offsets.data()), count * sizeof(uint64_t));
    out.write(reinterpret_cast<const char *>(store.lengths.data()), count * sizeof(uint64_t));
    const uint64_t size = store.bytes.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    out.write(reinterpret_cast<const char *>(store.bytes.data()), size);
    if (!out) throw std::runtime_error("Cannot write " + path);
}

} // namespace diskann::lvc
