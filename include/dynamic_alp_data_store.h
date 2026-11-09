#pragma once

#include <chrono>
#include <fstream>
#include <unordered_map>
#include "abstract_data_store.h"
#include "abstract_scratch.h"
#include "utils.h"

namespace diskann
{
// Existing legacy ALP records are immutable. New records use a chunked append
// arena, so growing the index does not relocate or re-encode the original data.
class DynamicALPDataStore : public AbstractDataStore<float>
{
    static constexpr uint64_t delta_bit = uint64_t(1) << 63;
    static constexpr size_t chunk_bytes = 4 * 1024 * 1024;
    struct AlignedVector
    {
        float *data = nullptr;
        explicit AlignedVector(size_t n)
        {
            alloc_aligned(reinterpret_cast<void **>(&data), n * sizeof(float), 32);
            std::fill(data, data + n, 0.0f);
        }
        ~AlignedVector() { aligned_free(data); }
        AlignedVector(const AlignedVector &) = delete;
        AlignedVector &operator=(const AlignedVector &) = delete;
    };
    std::unique_ptr<Distance<float>> _distance;
    size_t _aligned_dim, _initial_count = 0, _tail = chunk_bytes;
    bool _cache_enabled = true, _in_insert = false;
    std::vector<uint8_t> _base;
    std::vector<uint64_t> _offsets;
    std::vector<std::unique_ptr<uint8_t[]>> _chunks;
    mutable std::unordered_map<uint32_t, std::unique_ptr<AlignedVector>> _cache;
    mutable AlignedVector _first, _second;

    const uint8_t *record(uint32_t loc) const
    {
        if (loc >= _offsets.size() || _offsets[loc] == UINT64_MAX)
            throw std::out_of_range("Unoccupied ALP location");
        const uint64_t offset = _offsets[loc];
        const uint8_t *p;
        size_t available;
        if (offset & delta_bit)
        {
            uint64_t pos = offset & ~delta_bit;
            p = _chunks.at(pos / chunk_bytes).get() + pos % chunk_bytes;
            available = chunk_bytes - pos % chunk_bytes;
        }
        else
        {
            if (offset > _base.size()) throw std::out_of_range("ALP base offset");
            p = _base.data() + offset;
            available = _base.size() - offset;
        }
        if (available < 10 || p[0] > 16 || p[1] > 63 || 10 + (_dim * p[1] + 7) / 8 > available)
            throw std::runtime_error("Invalid ALP record boundary");
        return p;
    }
    void decode(uint32_t loc, float *out) const
    {
        const auto *p = record(loc);
        int64_t minimum;
        std::memcpy(&minimum, p + 2, 8);
        const double inv = std::pow(10.0, -int(p[0]));
        const unsigned width = p[1];
        size_t bit = 0;
        for (size_t d = 0; d < _dim; ++d)
        {
            uint64_t delta = 0;
            for (unsigned b = 0; b < width; ++b, ++bit)
                delta |= uint64_t((p[10 + bit / 8] >> (bit % 8)) & 1) << b;
            out[d] = float((minimum + int64_t(delta)) * inv);
        }
        std::fill(out + _dim, out + _aligned_dim, 0.0f);
    }
    const float *decoded(uint32_t loc, AlignedVector &scratch) const
    {
        if (_in_insert && _cache_enabled)
        {
            auto it = _cache.find(loc);
            if (it != _cache.end()) { ++cache_hits; return it->second->data; }
            auto value = std::make_unique<AlignedVector>(_aligned_dim);
            decode(loc, value->data);
            ++payload_decodes;
            auto *ptr = value->data;
            _cache.emplace(loc, std::move(value));
            cache_peak_nodes = std::max(cache_peak_nodes, uint64_t(_cache.size()));
            return ptr;
        }
        decode(loc, scratch.data);
        if (_in_insert) ++payload_decodes;
        return scratch.data;
    }

  public:
    uint64_t payload_bytes = 0, appended_records = 0;
    mutable uint64_t payload_decodes = 0, cache_hits = 0, cache_peak_nodes = 0, insert_distance_evaluations = 0;
    double encode_seconds = 0;
    DynamicALPDataStore(location_t capacity, size_t dim, std::unique_ptr<Distance<float>> distance)
        : AbstractDataStore<float>(capacity, dim), _distance(std::move(distance)),
          _aligned_dim(ROUND_UP(dim, _distance->get_required_alignment())),
          _offsets(capacity, UINT64_MAX), _first(_aligned_dim), _second(_aligned_dim) {}
    void set_insert_cache(bool value) { _cache_enabled = value; }
    void begin_insert() override { _cache.clear(); _in_insert = true; }
    void end_insert() override { _in_insert = false; _cache.clear(); }
    location_t load(const std::string &filename) override
    {
        if (_initial_count || !_chunks.empty()) throw std::runtime_error("ALP store already populated");
        std::ifstream in(filename, std::ios::binary);
        in.exceptions(std::ios::failbit | std::ios::badbit);
        uint32_t n, dim;
        uint64_t count, bytes;
        in.read(reinterpret_cast<char *>(&n), 4); in.read(reinterpret_cast<char *>(&dim), 4);
        in.read(reinterpret_cast<char *>(&count), 8);
        if (dim != _dim || !n || n > _capacity || count < n || count > _capacity)
            throw std::runtime_error("Expected a legacy ALP .data file with matching dimensions");
        in.read(reinterpret_cast<char *>(_offsets.data()), count * sizeof(uint64_t));
        std::fill(_offsets.begin() + n, _offsets.end(), UINT64_MAX);
        in.read(reinterpret_cast<char *>(&bytes), 8);
        const auto begin = in.tellg(); in.seekg(0, std::ios::end);
        if (uint64_t(in.tellg() - begin) != bytes) throw std::runtime_error("ALP file length mismatch");
        in.seekg(begin); _base.resize(bytes);
        in.read(reinterpret_cast<char *>(_base.data()), bytes);
        _initial_count = n; payload_bytes = bytes;
        for (uint32_t i = 0; i < n; ++i) (void)record(i);
        return n;
    }
    size_t save(const std::string &, location_t) override
    { throw std::logic_error("Dynamic ALP uses immutable base plus delta; compact save is disabled"); }
    size_t get_aligned_dim() const override { return _aligned_dim; }
    size_t get_alignment_factor() const override { return _distance->get_required_alignment(); }
    void populate_data(const float *, location_t) override { throw std::logic_error("Load an existing ALP index"); }
    void populate_data(const std::string &, size_t) override { throw std::logic_error("Load an existing ALP index"); }
    void extract_data_to_bin(const std::string &, location_t) override { throw std::logic_error("Not a build datastore"); }
    void get_vector(location_t loc, float *target) const override
    { std::memcpy(target, decoded(loc, _first), _dim * sizeof(float)); }
    void set_vector(location_t loc, const float *const values) override
    {
        if (loc >= _capacity || _offsets[loc] != UINT64_MAX)
            throw std::runtime_error("ALP insertion requires an unused physical location");
        const auto started = std::chrono::steady_clock::now();
        int exp = -1;
        double factor = 1;
        for (int e = 0; e <= 16; ++e)
        {
            factor = std::pow(10.0, e); bool valid = true;
            for (size_t d = 0; d < _dim; ++d)
                if (!std::isfinite(values[d]) || std::abs(double(values[d]) * factor) >= 9.22e18 ||
                    std::abs(double(values[d]) - std::round(double(values[d]) * factor) * (1.0 / factor)) > 1e-4)
                { valid = false; break; }
            if (valid) { exp = e; break; }
        }
        if (exp < 0) throw std::runtime_error("ALP value exceeds supported integer scaling range");
        std::vector<int64_t> ints(_dim);
        for (size_t d = 0; d < _dim; ++d) ints[d] = std::llround(double(values[d]) * factor);
        const auto extrema = std::minmax_element(ints.begin(), ints.end());
        const int64_t minimum = *extrema.first;
        uint64_t range = uint64_t(*extrema.second) - uint64_t(minimum);
        unsigned width = 0; for (auto v = range; v; v >>= 1) ++width;
        if (width > 63) throw std::runtime_error("ALP delta overflow");
        std::vector<uint8_t> bytes(10 + (_dim * width + 7) / 8, 0);
        bytes[0] = uint8_t(exp); bytes[1] = uint8_t(width);
        std::memcpy(bytes.data() + 2, &minimum, 8);
        size_t bit = 0;
        for (auto value : ints)
        {
            uint64_t delta = uint64_t(value) - uint64_t(minimum);
            for (unsigned b = 0; b < width; ++b, ++bit)
                bytes[10 + bit / 8] |= uint8_t((delta >> b) & 1) << (bit % 8);
        }
        if (bytes.size() > chunk_bytes) throw std::runtime_error("ALP record exceeds chunk size");
        if (_tail + bytes.size() > chunk_bytes) { _chunks.emplace_back(new uint8_t[chunk_bytes]); _tail = 0; }
        const uint64_t offset = (_chunks.size() - 1) * chunk_bytes + _tail;
        std::memcpy(_chunks.back().get() + _tail, bytes.data(), bytes.size());
        _offsets[loc] = delta_bit | offset; _tail += bytes.size();
        payload_bytes += bytes.size(); ++appended_records;
        if (_in_insert) encode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    void prefetch_vector(location_t loc) override { diskann::prefetch_vector(reinterpret_cast<const char *>(record(loc)), 10); }
    void copy_vectors(location_t from, location_t to, location_t count) override
    {
        if (from + uint64_t(count) > _capacity || to + uint64_t(count) > _capacity)
            throw std::out_of_range("ALP copy range");
        std::memmove(_offsets.data() + to, _offsets.data() + from, count * sizeof(uint64_t));
    }
    void move_vectors(location_t from, location_t to, location_t count) override
    {
        copy_vectors(from, to, count);
        for (uint64_t i = from; i < uint64_t(from) + count; ++i)
            if (i < to || i >= uint64_t(to) + count) _offsets[i] = UINT64_MAX;
    }
    void preprocess_query(const float *query, AbstractScratch<float> *scratch) const override
    {
        if (!scratch) throw std::runtime_error("Missing query scratch");
        std::memmove(scratch->aligned_query_T(), query, _dim * sizeof(float));
        std::fill(scratch->aligned_query_T() + _dim, scratch->aligned_query_T() + _aligned_dim, 0.0f);
    }
    float get_distance(const float *query, location_t loc) const override
    {
        if (_in_insert) ++insert_distance_evaluations;
        return _distance->compare(query, decoded(loc, _first), uint32_t(_aligned_dim));
    }
    float get_distance(location_t a, location_t b) const override
    {
        if (_in_insert) ++insert_distance_evaluations;
        const auto *first = decoded(a, _first); const auto *second = decoded(b, _second);
        return _distance->compare(first, second, uint32_t(_aligned_dim));
    }
    void get_distance(const float *q, const location_t *locs, uint32_t n, float *out, AbstractScratch<float> *) const override
    { for (uint32_t i = 0; i < n; ++i) out[i] = get_distance(q, locs[i]); }
    void get_distance(const float *q, const std::vector<location_t> &locs, std::vector<float> &out, AbstractScratch<float> *) const override
    { out.resize(locs.size()); for (size_t i = 0; i < locs.size(); ++i) out[i] = get_distance(q, locs[i]); }
    location_t calculate_medoid() const override { throw std::logic_error("Import the existing graph entry point"); }
    Distance<float> *get_dist_fn() const override { return _distance.get(); }
    uint64_t payload_capacity_bytes() const { return _base.capacity() + _chunks.size() * chunk_bytes; }
    uint64_t metadata_bytes() const { return _offsets.capacity() * sizeof(uint64_t); }
  protected:
    location_t expand(location_t n) override { _offsets.resize(n, UINT64_MAX); _capacity = n; return n; }
    location_t shrink(location_t) override { throw std::logic_error("Dynamic ALP compaction is disabled"); }
};
} // namespace diskann
