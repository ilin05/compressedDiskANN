#pragma once

#include <fstream>
#include <limits>
#include "pq.h"
#include "pq_common.h"

namespace diskann
{
// Serial dynamic-update sidecar, deliberately independent of pq_dist_build.
class DynamicVamanaPQ
{
    FixedChunkPQTable _table;
    std::vector<uint8_t> _codes;
    size_t _chunks, _dim, _capacity;

  public:
    mutable uint64_t distance_evaluations = 0;
    DynamicVamanaPQ(const DynamicVamanaPQ &) = delete;
    DynamicVamanaPQ &operator=(const DynamicVamanaPQ &) = delete;
    DynamicVamanaPQ(const std::string &prefix, size_t chunks, size_t dim, size_t capacity, size_t expected)
        : _chunks(chunks), _dim(dim), _capacity(capacity)
    {
        if (!chunks || chunks > dim || capacity > SIZE_MAX / chunks)
            throw std::runtime_error("Invalid dynamic PQ dimensions/capacity");
        _table.load_pq_centroid_bin(get_pivot_data_filename(prefix, false, uint32_t(chunks)).c_str(), chunks);
        if (_table.get_dimension() != dim) throw std::runtime_error("Dynamic PQ pivot dimension mismatch");
        std::ifstream input(get_quantized_vectors_filename(prefix, false, uint32_t(chunks)), std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        uint32_t n, width;
        input.read(reinterpret_cast<char *>(&n), 4);
        input.read(reinterpret_cast<char *>(&width), 4);
        if (n != expected || width != chunks || n > capacity)
            throw std::runtime_error("Dynamic PQ rows/width do not match the initial payloads");
        _codes.resize(capacity * chunks, 0);
        input.read(reinterpret_cast<char *>(_codes.data()), size_t(n) * chunks);
        if (input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Unexpected trailing PQ codes");
    }
    void set_vector(uint32_t loc, const float *vector)
    {
        if (loc >= _capacity) throw std::out_of_range("Dynamic PQ location");
        _table.encode_vector(vector, _codes.data() + size_t(loc) * _chunks);
    }
    void copy_vector(uint32_t source, uint32_t target)
    {
        if (source >= _capacity || target >= _capacity) throw std::out_of_range("Dynamic PQ copy");
        std::memmove(_codes.data() + size_t(target) * _chunks,
                     _codes.data() + size_t(source) * _chunks, _chunks);
    }
    void prepare(const float *query, std::vector<float> &lut)
    {
        std::vector<float> copy(query, query + _dim);
        _table.preprocess_query(copy.data());
        lut.resize(256 * _chunks);
        _table.populate_chunk_distances(copy.data(), lut.data());
    }
    float distance(uint32_t loc, const std::vector<float> &lut) const
    {
        if (loc >= _capacity) throw std::out_of_range("Dynamic PQ distance");
        ++distance_evaluations;
        float result = 0;
        for (size_t c = 0; c < _chunks; ++c)
            result += lut[c * 256 + _codes[size_t(loc) * _chunks + c]];
        return result;
    }
    bool verify(uint32_t loc, const float *vector)
    {
        std::vector<uint8_t> encoded(_chunks);
        _table.encode_vector(vector, encoded.data());
        return std::equal(encoded.begin(), encoded.end(), _codes.begin() + size_t(loc) * _chunks);
    }
    uint64_t bytes() const { return _codes.size() + _table.accounted_memory_bytes(); }
};
} // namespace diskann
