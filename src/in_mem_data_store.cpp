// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <memory>
#include <type_traits>
#include "abstract_scratch.h"
#include "in_mem_data_store.h"
#include "lvc_vamana_file.h"

#include "utils.h"

namespace diskann
{

template <typename T> struct LvcDecodeScratch
{
    T *first = nullptr;
    T *second = nullptr;
    size_t capacity = 0;

    ~LvcDecodeScratch()
    {
        aligned_free(first);
        aligned_free(second);
    }

    void ensure(size_t dim)
    {
        if (dim <= capacity) return;
        aligned_free(first);
        aligned_free(second);
        first = second = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&first), dim * sizeof(T), 32);
        alloc_aligned(reinterpret_cast<void **>(&second), dim * sizeof(T), 32);
        capacity = dim;
    }
};

template <typename T> LvcDecodeScratch<T> &lvc_decode_scratch()
{
    thread_local LvcDecodeScratch<T> scratch;
    return scratch;
}


struct BitWriter {
    std::vector<uint8_t>& buf;
    uint8_t cur_byte = 0;
    int bit_pos = 0;

    BitWriter(std::vector<uint8_t>& b) : buf(b) {}
    void write(uint64_t val, int bits) {
        for (int i = 0; i < bits; ++i) {
            uint8_t b = (val >> i) & 1;
            cur_byte |= (b << bit_pos);
            bit_pos++;
            if (bit_pos == 8) {
                buf.push_back(cur_byte);
                cur_byte = 0;
                bit_pos = 0;
            }
        }
    }
    void flush() {
        if (bit_pos > 0) {
            buf.push_back(cur_byte);
            cur_byte = 0;
            bit_pos = 0;
        }
    }
};

struct BitReader {
    const uint8_t* buf;
    size_t byte_pos = 0;
    int bit_pos = 0;

    BitReader(const uint8_t* b) : buf(b) {}
    uint64_t read(int bits) {
        uint64_t val = 0;
        for (int i = 0; i < bits; ++i) {
            uint64_t b = (buf[byte_pos] >> bit_pos) & 1;
            val |= (b << i);
            bit_pos++;
            if (bit_pos == 8) {
                byte_pos++;
                bit_pos = 0;
            }
        }
        return val;
    }
};

inline int get_bit_width(uint64_t n) {
    if (n == 0) return 0;
    int bits = 0;
    while (n > 0) {
        n >>= 1;
        bits++;
    }
    return bits;
}

template <typename data_t>
void InMemDataStore<data_t>::encode_and_store(const data_t *uncompressed_vec, location_t loc)
{
    std::vector<uint8_t> temp;
    BitWriter writer(temp);

    if constexpr (std::is_same_v<data_t, float>) {
        int selected_exp = 16;
        for (int exp = 0; exp <= 16; ++exp) {
            double factor = std::pow(10.0, exp);
            double inv_factor = 1.0 / factor;
            bool ok = true;
            for (size_t i = 0; i < this->_dim; ++i) {
                double origin = uncompressed_vec[i];
                double reconstructed = std::round(origin * factor) * inv_factor;
                if (std::abs(origin - reconstructed) > 1e-4) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                selected_exp = exp;
                break;
            }
        }
        writer.write((uint64_t)selected_exp, 8); 
        double factor = std::pow(10.0, selected_exp);
        int64_t min_val = INT64_MAX;
        int64_t max_val = INT64_MIN;
        for (size_t i = 0; i < this->_dim; ++i) {
            int64_t val = (int64_t)std::llround(uncompressed_vec[i] * factor);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        uint64_t range = (uint64_t)(max_val - min_val);
        int bit_width = get_bit_width(range);
        writer.write((uint64_t)bit_width, 8);
        writer.write((uint64_t)min_val, 64);
        for (size_t i = 0; i < this->_dim; ++i) {
            int64_t val = (int64_t)std::llround(uncompressed_vec[i] * factor);
            writer.write((uint64_t)(val - min_val), bit_width);
        }
    } else {
        int64_t min_val = INT64_MAX;
        int64_t max_val = INT64_MIN;
        for (size_t i = 0; i < this->_dim; ++i) {
            int64_t val = (int64_t)uncompressed_vec[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        uint64_t range = (uint64_t)(max_val - min_val);
        int bit_width = get_bit_width(range);
        writer.write((uint64_t)bit_width, 8);
        writer.write((uint64_t)min_val, 64);
        for (size_t i = 0; i < this->_dim; ++i) {
            int64_t val = (int64_t)uncompressed_vec[i];
            writer.write((uint64_t)(val - min_val), bit_width);
        }
    }
    writer.flush();

    // Store offsets and append to compressed data
    if (_vector_offsets.size() <= loc)
    {
        _vector_offsets.resize(loc + 1, 0);
    }
    size_t current_offset = _compressed_data.size();
    _vector_offsets[loc] = current_offset;
    _compressed_data.insert(_compressed_data.end(), temp.begin(), temp.end());
}

template <typename data_t>
void InMemDataStore<data_t>::decode_vector(location_t loc, data_t *out_vec) const
{
    if (_lvc_mode != 0)
    {
        if constexpr (!std::is_same_v<data_t, float>)
        {
            throw std::runtime_error("LVC Vamana payload requires F32");
        }
        else
        {
            if (_lvc_mode == 1)
            {
                const size_t offset = size_t(loc) * this->_dim * sizeof(float);
                if (loc >= this->capacity() || offset > _compressed_data.size() ||
                    this->_dim * sizeof(float) > _compressed_data.size() - offset)
                    throw std::runtime_error("Raw LVC vector out of bounds");
                std::memcpy(out_vec, _compressed_data.data() + offset, this->_dim * sizeof(float));
            }
            else if (_lvc_mode == 2)
                decode_lvc_record<hnswlib::codecs::DeXORCodecPolicy>(loc, out_vec);
            else if (_lvc_mode == 3)
                decode_lvc_record<hnswlib::codecs::GorillaCodecPolicy>(loc, out_vec);
            else if (_lvc_mode == 4)
                decode_lvc_record<hnswlib::codecs::ElfCodecPolicy>(loc, out_vec);
            else
                throw std::runtime_error("Unknown LVC codec");
            std::fill(out_vec + this->_dim, out_vec + _aligned_dim, data_t{});
            return;
        }
    }
    size_t offset = _vector_offsets[loc];
    const uint8_t* compressed_ptr = &_compressed_data[offset];
    BitReader reader(compressed_ptr);

    if constexpr (std::is_same_v<data_t, float>) {
        int exp = reader.read(8);
        int bit_width = reader.read(8);
        int64_t min_val = (int64_t)reader.read(64);
        double factor = std::pow(10.0, -exp);
        for (size_t i = 0; i < this->_dim; ++i) {
            int64_t delta = (int64_t)reader.read(bit_width);
            int64_t val = min_val + delta;
            out_vec[i] = static_cast<float>(val * factor);
        }
        // Zero out the remaining aligned dimensions
        for (size_t i = this->_dim; i < this->_aligned_dim; ++i) {
            out_vec[i] = 0;
        }
    } else {
        int bit_width = reader.read(8);
        int64_t min_val = (int64_t)reader.read(64);
        for (size_t i = 0; i < this->_dim; ++i) {
            int64_t delta = (int64_t)reader.read(bit_width);
            int64_t val = min_val + delta;
            out_vec[i] = static_cast<data_t>(val);
        }
        for (size_t i = this->_dim; i < this->_aligned_dim; ++i) {
            out_vec[i] = 0;
        }
    }
}

template <typename data_t>
template <typename Codec>
void InMemDataStore<data_t>::decode_lvc_record(location_t loc, data_t *out_vec) const
{
    if constexpr (std::is_same_v<data_t, float>)
    {
        if (loc >= _lvc_parent.size()) throw std::runtime_error("LVC node out of bounds");
        std::vector<typename Codec::StateType> states(this->_dim);
        const lvc::RootStateCache<Codec> *cache = nullptr;
        if constexpr (std::is_same_v<Codec, hnswlib::codecs::DeXORCodecPolicy>) cache = _dexor_state_cache.get();
        else if constexpr (std::is_same_v<Codec, hnswlib::codecs::GorillaCodecPolicy>) cache = _gorilla_state_cache.get();
        else if constexpr (std::is_same_v<Codec, hnswlib::codecs::ElfCodecPolicy>) cache = _elf_state_cache.get();
        auto decode_one = [&](location_t id, data_t *target) {
            const size_t offset = _vector_offsets[id];
            const uint64_t length = _lvc_lengths[id];
            if (offset > _compressed_data.size() || length > _compressed_data.size() - offset)
                throw std::runtime_error("LVC record out of bounds");
            lvc::decode_record<Codec>(_compressed_data.data() + offset, length, states, target);
        };
        const uint32_t root = _lvc_parent[loc];
        const auto *entry = cache == nullptr ? nullptr : cache->find(root == lvc::no_parent ? loc : root);
        if (root == lvc::no_parent && entry != nullptr) {
            std::copy(entry->vector.begin(), entry->vector.end(), out_vec);
            return;
        }
        if (entry != nullptr) states = entry->states;
        else if (root != lvc::no_parent) decode_one(root, nullptr);
        decode_one(loc, out_vec);
    }
}

template <typename data_t>
std::vector<uint32_t> InMemDataStore<data_t>::lvc_roots() const
{
    std::vector<uint32_t> roots;
    for (uint32_t id = 0; id < _lvc_parent.size(); ++id)
        if (_lvc_parent[id] == lvc::no_parent) roots.push_back(id);
    return roots;
}

template <typename data_t>
void InMemDataStore<data_t>::load_lvc_state_cache(const std::vector<uint32_t> &roots)
{
    _dexor_state_cache.reset();
    _gorilla_state_cache.reset();
    _elf_state_cache.reset();
    if (roots.empty() || _lvc_mode < 2) return;
    auto add_roots = [&](auto &cache) {
        cache.reserve(roots.size());
        for (uint32_t id : roots) {
            if (id >= _lvc_parent.size() || _lvc_parent[id] != lvc::no_parent)
                throw std::runtime_error("LVC state cache requires root IDs");
            const size_t offset = _vector_offsets[id];
            const uint64_t length = _lvc_lengths[id];
            if (offset > _compressed_data.size() || length > _compressed_data.size() - offset)
                throw std::runtime_error("LVC root record out of bounds");
            cache.insert(id, _compressed_data.data() + offset, length, this->_dim);
        }
    };
    if (_lvc_mode == 2) {
        _dexor_state_cache = std::make_unique<lvc::RootStateCache<hnswlib::codecs::DeXORCodecPolicy>>();
        add_roots(*_dexor_state_cache);
        diskann::cout << "LVC Vamana state cache roots=" << _dexor_state_cache->size()
                      << " payload_bytes=" << _dexor_state_cache->payload_bytes() << std::endl;
    } else if (_lvc_mode == 3) {
        _gorilla_state_cache = std::make_unique<lvc::RootStateCache<hnswlib::codecs::GorillaCodecPolicy>>();
        add_roots(*_gorilla_state_cache);
        diskann::cout << "LVC Vamana state cache roots=" << _gorilla_state_cache->size()
                      << " payload_bytes=" << _gorilla_state_cache->payload_bytes() << std::endl;
    } else if (_lvc_mode == 4) {
        _elf_state_cache = std::make_unique<lvc::RootStateCache<hnswlib::codecs::ElfCodecPolicy>>();
        add_roots(*_elf_state_cache);
        diskann::cout << "LVC Vamana state cache roots=" << _elf_state_cache->size()
                      << " payload_bytes=" << _elf_state_cache->payload_bytes() << std::endl;
    }
}

template <typename data_t>
InMemDataStore<data_t>::InMemDataStore(const location_t num_points, const size_t dim,
                                       std::unique_ptr<Distance<data_t>> distance_fn)
    : AbstractDataStore<data_t>(num_points, dim), _distance_fn(std::move(distance_fn))
{
    _aligned_dim = ROUND_UP(dim, _distance_fn->get_required_alignment());
    _vector_offsets.reserve(this->_capacity + 1);
}

template <typename data_t> InMemDataStore<data_t>::~InMemDataStore()
{
    // Resources in std::vector will be freed automatically
}

template <typename data_t> size_t InMemDataStore<data_t>::get_aligned_dim() const
{
    return _aligned_dim;
}

template <typename data_t> size_t InMemDataStore<data_t>::get_alignment_factor() const
{
    return _distance_fn->get_required_alignment();
}

template <typename data_t> location_t InMemDataStore<data_t>::load(const std::string &filename)
{
    load_lvc_state_cache({});
    return load_impl(filename);
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t InMemDataStore<data_t>::load_impl(AlignedFileReader &reader)
{
    size_t file_dim, file_num_points;

    diskann::get_bin_metadata(reader, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension," << "but file has " << file_dim
               << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }
    copy_aligned_data_from_file<data_t>(reader, _data, file_num_points, file_dim, _aligned_dim);

    return (location_t)file_num_points;
}
#endif

template <typename data_t> location_t InMemDataStore<data_t>::load_impl(const std::string &filename)
{
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open())
        throw diskann::ANNException("ERROR: Could not open " + filename + " for reading", -1);

    uint32_t file_num_points;
    uint32_t file_dim;
    reader.read((char*)&file_num_points, sizeof(uint32_t));
    reader.read((char*)&file_dim, sizeof(uint32_t));

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension," << "but file has " << file_dim
               << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }

    uint64_t magic = 0;
    reader.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (magic == lvc::vamana_magic)
    {
        uint32_t version = 0;
        reader.read(reinterpret_cast<char *>(&version), sizeof(version));
        reader.read(reinterpret_cast<char *>(&_lvc_mode), sizeof(_lvc_mode));
        if (version != lvc::vamana_version || _lvc_mode < 1 || _lvc_mode > 4 ||
            !std::is_same_v<data_t, float>)
            throw std::runtime_error("Unsupported LVC Vamana data format");
        if (_lvc_mode != 1)
        {
            _lvc_parent.resize(file_num_points);
            _lvc_depth.resize(file_num_points);
            reader.read(reinterpret_cast<char *>(_lvc_parent.data()), file_num_points * sizeof(uint32_t));
            reader.read(reinterpret_cast<char *>(_lvc_depth.data()), file_num_points);
            std::vector<uint64_t> offsets(file_num_points);
            _lvc_lengths.resize(file_num_points);
            reader.read(reinterpret_cast<char *>(offsets.data()), file_num_points * sizeof(uint64_t));
            reader.read(reinterpret_cast<char *>(_lvc_lengths.data()), file_num_points * sizeof(uint64_t));
            _vector_offsets.assign(offsets.begin(), offsets.end());
            lvc::Forest forest;
            forest.parent = _lvc_parent;
            forest.depth.assign(_lvc_depth.begin(), _lvc_depth.end());
            forest.initial_roots = static_cast<uint32_t>(std::count(_lvc_parent.begin(), _lvc_parent.end(), lvc::no_parent));
            lvc::validate_forest(forest);
        }
        uint64_t payload_size = 0;
        reader.read(reinterpret_cast<char *>(&payload_size), sizeof(payload_size));
        const auto payload_begin = reader.tellg();
        reader.seekg(0, std::ios::end);
        const auto payload_end = reader.tellg();
        if (payload_begin == std::streampos(-1) || payload_end == std::streampos(-1) ||
            payload_end < payload_begin ||
            uint64_t(payload_end - payload_begin) != payload_size)
            throw std::runtime_error("Invalid LVC payload file length");
        reader.seekg(payload_begin);
        if (_lvc_mode == 1 && payload_size != uint64_t(file_num_points) * this->_dim * sizeof(float))
            throw std::runtime_error("Invalid raw LVC payload length");
        _compressed_data.resize(payload_size);
        reader.read(reinterpret_cast<char *>(_compressed_data.data()), payload_size);
        if (!reader) throw std::runtime_error("Truncated LVC Vamana data file");
        if (_lvc_mode != 1)
            for (size_t i = 0; i < file_num_points; ++i)
                if (_vector_offsets[i] > payload_size || _lvc_lengths[i] < 8 ||
                    _lvc_lengths[i] > payload_size - _vector_offsets[i])
                    throw std::runtime_error("Invalid LVC record boundary");
        return (location_t)file_num_points;
    }
    reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
    _lvc_mode = 0;
    size_t num_offsets;
    reader.read((char*)&num_offsets, sizeof(size_t));
    _vector_offsets.resize(num_offsets);
    if (num_offsets > 0) {
        reader.read((char*)_vector_offsets.data(), num_offsets * sizeof(size_t));
    }

    size_t comp_size;
    reader.read((char*)&comp_size, sizeof(size_t));
    _compressed_data.resize(comp_size);
    if (comp_size > 0) {
        reader.read((char*)_compressed_data.data(), comp_size * sizeof(uint8_t));
    }

    reader.close();
    return (location_t)file_num_points;
}

template <typename data_t> size_t InMemDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    std::ofstream writer(filename, std::ios::binary);
    if (!writer.is_open())
        throw diskann::ANNException("ERROR: Could not open " + filename + " for writing", -1);
    
    // Header
    uint32_t npts = num_points;
    uint32_t dim_dummy = this->_dim;
    writer.write((char*)&npts, sizeof(uint32_t));
    writer.write((char*)&dim_dummy, sizeof(uint32_t));

    if (_lvc_mode != 0)
    {
        writer.write(reinterpret_cast<const char *>(&lvc::vamana_magic), sizeof(lvc::vamana_magic));
        writer.write(reinterpret_cast<const char *>(&lvc::vamana_version), sizeof(lvc::vamana_version));
        writer.write(reinterpret_cast<const char *>(&_lvc_mode), sizeof(_lvc_mode));
        if (_lvc_mode != 1)
        {
            writer.write(reinterpret_cast<const char *>(_lvc_parent.data()), num_points * sizeof(uint32_t));
            writer.write(reinterpret_cast<const char *>(_lvc_depth.data()), num_points);
            std::vector<uint64_t> offsets(_vector_offsets.begin(), _vector_offsets.begin() + num_points);
            writer.write(reinterpret_cast<const char *>(offsets.data()), num_points * sizeof(uint64_t));
            writer.write(reinterpret_cast<const char *>(_lvc_lengths.data()), num_points * sizeof(uint64_t));
        }
        const uint64_t payload_size = _compressed_data.size();
        writer.write(reinterpret_cast<const char *>(&payload_size), sizeof(payload_size));
        writer.write(reinterpret_cast<const char *>(_compressed_data.data()), payload_size);
        return num_points;
    }

    // offsets table
    size_t num_offsets = _vector_offsets.size();
    writer.write((char*)&num_offsets, sizeof(size_t));
    if (num_offsets > 0) {
        writer.write((char*)_vector_offsets.data(), num_offsets * sizeof(size_t));
    }

    // compressed data
    size_t comp_size = _compressed_data.size();
    writer.write((char*)&comp_size, sizeof(size_t));
    if (comp_size > 0) {
        writer.write((char*)_compressed_data.data(), comp_size * sizeof(uint8_t));
    }

    writer.close();
    return num_points;
}

template <typename data_t> void InMemDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    _lvc_mode = 0;
    // Instead of copying to _data directly, encode each vector
    _compressed_data.clear();
    _vector_offsets.clear();
    _vector_offsets.resize(num_pts, 0);

    for (location_t i = 0; i < num_pts; i++)
    {
        // Extract row i to encode
        encode_and_store(vectors + i * this->_dim, i);
    }
}

template <typename data_t> void InMemDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    _lvc_mode = 0;
    size_t npts, ndim;
    
    // Check first to allocate properly instead of massive copy
    diskann::get_bin_metadata(filename, npts, ndim);

    if ((location_t)npts > this->capacity())
    {
        std::stringstream ss;
        ss << "Number of points in the file: " << filename
           << " is greater than the capacity of data store: " << this->capacity()
           << ". Must invoke resize before calling populate_data()" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if ((location_t)ndim != this->get_dims())
    {
        std::stringstream ss;
        ss << "Number of dimensions of a point in the file: " << filename
           << " is not equal to dimensions of data store: " << this->capacity() << "." << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    _compressed_data.clear();
    _vector_offsets.clear();
    _vector_offsets.resize(npts, 0);

    data_t* temp_vec;
    alloc_aligned(((void **)&temp_vec), _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(temp_vec, 0, _aligned_dim * sizeof(data_t));

    std::ifstream reader(filename, std::ios::binary);
    reader.seekg(offset + 8, std::ios::beg); // Skip header

    for (size_t i = 0; i < npts; ++i) {
        reader.read((char*)temp_vec, ndim * sizeof(data_t));
        std::memset(temp_vec + ndim, 0, (_aligned_dim - ndim) * sizeof(data_t));
        encode_and_store(temp_vec, i);
    }

    aligned_free(temp_vec);
    reader.close();
}

template <typename data_t>
void InMemDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_points)
{
    std::ofstream writer(filename, std::ios::binary);
    if (!writer.is_open())
        throw diskann::ANNException("ERROR: Could not open " + filename + " for writing", -1);

    uint32_t npts = num_points;
    uint32_t ndim = this->_dim;
    writer.write((char*)&npts, sizeof(uint32_t));
    writer.write((char*)&ndim, sizeof(uint32_t));

    data_t* temp_vec;
    alloc_aligned(((void **)&temp_vec), _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(temp_vec, 0, _aligned_dim * sizeof(data_t));

    for (location_t i = 0; i < num_points; ++i) {
        decode_vector(i, temp_vec);
        writer.write((char*)temp_vec, ndim * sizeof(data_t));
    }

    aligned_free(temp_vec);
    writer.close();
}

template <typename data_t> void InMemDataStore<data_t>::get_vector(const location_t i, data_t *dest) const
{
    // REFACTOR TODO: Should we denormalize and return values?
    std::vector<data_t> temp_dec(this->_aligned_dim);
    decode_vector(i, temp_dec.data());
    memcpy(dest, temp_dec.data(), this->_dim * sizeof(data_t));
}

template <typename data_t> void InMemDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    if (_lvc_mode != 0) throw std::runtime_error("D2 LVC Vamana supports static payloads only");
    // Need to handle resizing dynamically or preallocate in a real impl. Not strictly required for purely static build
    std::vector<data_t> aligned_vec(this->_aligned_dim, 0);
    memcpy(aligned_vec.data(), vector, this->_dim * sizeof(data_t));
    encode_and_store(aligned_vec.data(), loc);
}

template <typename data_t> void InMemDataStore<data_t>::prefetch_vector(const location_t loc)
{
    if (_lvc_mode == 1)
    {
        const size_t offset = size_t(loc) * this->_dim * sizeof(data_t);
        if (offset < _compressed_data.size())
            diskann::prefetch_vector((const char *)(_compressed_data.data() + offset), this->_dim * sizeof(data_t));
        return;
    }
    if (loc < _vector_offsets.size()) {
        size_t offset = _vector_offsets[loc];
        // Prefetch the compressed data rather than raw block
        diskann::prefetch_vector((const char *)&_compressed_data[offset], sizeof(uint8_t) * _aligned_dim);
    }
}

template <typename data_t>
void InMemDataStore<data_t>::preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const
{
    if (query_scratch != nullptr)
    {
        memcpy(query_scratch->aligned_query_T(), query, sizeof(data_t) * this->get_dims());
    }
    else
    {
        std::stringstream ss;
        ss << "In InMemDataStore::preprocess_query: Query scratch is null";
        diskann::cerr << ss.str() << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
}

template <typename data_t> float InMemDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    if (_lvc_mode != 0)
    {
        auto &scratch = lvc_decode_scratch<data_t>();
        scratch.ensure(_aligned_dim);
        decode_vector(loc, scratch.first);
        return _distance_fn->compare(query, scratch.first, (uint32_t)_aligned_dim);
    }
    // WARNING: dynamically allocating buffer per distance calc is slow. Kept for minimal architecture validation
    // Future performance patches would place temp_dec on AbstractScratch.
    std::vector<data_t> temp_dec(this->_aligned_dim);
    decode_vector(loc, temp_dec.data());
    return _distance_fn->compare(query, temp_dec.data(), (uint32_t)_aligned_dim);
}

template <typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *query, const location_t *locations,
                                          const uint32_t location_count, float *distances,
                                          AbstractScratch<data_t> *scratch_space) const
{
    if (_lvc_mode != 0)
    {
        auto &local = lvc_decode_scratch<data_t>();
        local.ensure(_aligned_dim);
        for (location_t i = 0; i < location_count; ++i)
        {
            decode_vector(locations[i], local.first);
            distances[i] = _distance_fn->compare(query, local.first, (uint32_t)_aligned_dim);
        }
        return;
    }
    std::vector<data_t> temp_dec(this->_aligned_dim);
    for (location_t i = 0; i < location_count; i++)
    {
        decode_vector(locations[i], temp_dec.data());
        distances[i] = _distance_fn->compare(query, temp_dec.data(), (uint32_t)this->_aligned_dim);
    }
}

template <typename data_t>
float InMemDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    if (_lvc_mode != 0)
    {
        auto &scratch = lvc_decode_scratch<data_t>();
        scratch.ensure(_aligned_dim);
        decode_vector(loc1, scratch.first);
        decode_vector(loc2, scratch.second);
        return _distance_fn->compare(scratch.first, scratch.second, (uint32_t)_aligned_dim);
    }
    std::vector<data_t> temp_dec1(this->_aligned_dim);
    std::vector<data_t> temp_dec2(this->_aligned_dim);
    decode_vector(loc1, temp_dec1.data());
    decode_vector(loc2, temp_dec2.data());
    return _distance_fn->compare(temp_dec1.data(), temp_dec2.data(), (uint32_t)this->_aligned_dim);
}

template <typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                          std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const
{
    if (_lvc_mode != 0)
    {
        auto &local = lvc_decode_scratch<data_t>();
        local.ensure(_aligned_dim);
        for (size_t i = 0; i < ids.size(); ++i)
        {
            decode_vector(ids[i], local.first);
            distances[i] = _distance_fn->compare(preprocessed_query, local.first, (uint32_t)_aligned_dim);
        }
        return;
    }
    std::vector<data_t> temp_dec(this->_aligned_dim);
    for (int i = 0; i < ids.size(); i++)
    {
        decode_vector(ids[i], temp_dec.data());
        distances[i] =
            _distance_fn->compare(preprocessed_query, temp_dec.data(), (uint32_t)this->_aligned_dim);
    }
}

template <typename data_t> location_t InMemDataStore<data_t>::expand(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size < this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
    _vector_offsets.resize(new_size, 0);
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t> location_t InMemDataStore<data_t>::shrink(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size > this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
    _vector_offsets.resize(new_size);
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t>
void InMemDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                          const location_t num_locations)
{
    if (_lvc_mode != 0) throw std::runtime_error("D2 LVC Vamana supports static payloads only");
    if (num_locations == 0 || old_location_start == new_location_start) return;
    // For pure static tests without dynamically sized rewrites, we can just alter offsets.
    // If elements have actual variable size and are moved around in the compressed buffer, 
    // it requires appending to _compressed_data or garbage collection.
    // Here we just shuffle offset mapping for simplicity.
    if (new_location_start > old_location_start) {
        for (long long i = num_locations - 1; i >= 0; --i) {
            _vector_offsets[new_location_start + i] = _vector_offsets[old_location_start + i];
        }
    } else {
        for (long long i = 0; i < num_locations; ++i) {
            _vector_offsets[new_location_start + i] = _vector_offsets[old_location_start + i];
        }
    }
}

template <typename data_t>
void InMemDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
                                          const location_t num_points)
{
    if (_lvc_mode != 0) throw std::runtime_error("D2 LVC Vamana supports static payloads only");
    assert(from_loc < this->_capacity);
    assert(to_loc < this->_capacity);
    assert(num_points < this->_capacity);
    for (location_t i = 0; i < num_points; ++i) {
        _vector_offsets[to_loc + i] = _vector_offsets[from_loc + i];
    }
}

template <typename data_t> location_t InMemDataStore<data_t>::calculate_medoid() const
{
    // allocate and init centroid
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    std::vector<data_t> temp_dec(_aligned_dim);
    for (size_t i = 0; i < this->capacity(); i++) {
        decode_vector(i, temp_dec.data());
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += (float)temp_dec[j];
    }

    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)this->capacity();

    // compute all to one distance
    float *distances = new float[this->capacity()];

    // TODO: REFACTOR. Removing pragma might make this slow. Must revisit.
    //  Problem is that we need to pass num_threads here, it is not clear
    //  if data store must be aware of threads!
    // #pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t)this->capacity(); i++)
    {
        // extract point and distance reference
        float &dist = distances[i];
        
        std::vector<data_t> temp_dec_2(_aligned_dim);
        decode_vector(i, temp_dec_2.data());
        const data_t *cur_vec = temp_dec_2.data();
        
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (float)cur_vec[j]) * (center[j] - (float)cur_vec[j]);
            dist += diff;
        }
    }
    // find imin
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < this->capacity(); i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

template <typename data_t> Distance<data_t> *InMemDataStore<data_t>::get_dist_fn() const
{
    return this->_distance_fn.get();
}

template DISKANN_DLLEXPORT class InMemDataStore<float>;
template DISKANN_DLLEXPORT class InMemDataStore<int8_t>;
template DISKANN_DLLEXPORT class InMemDataStore<uint8_t>;

} // namespace diskann
