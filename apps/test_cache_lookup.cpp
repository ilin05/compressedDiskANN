// Lookup-only capacity profiling; no full-precision payload load or forest build.
#include "common_includes.h"
#include "pq_flash_index.h"
#include "cache_lookup_probe.h"
#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

int main(int argc, char **argv) {
    try {
        std::string index_prefix, ids_path, csv_path;
        uint64_t maximum = 0, limit = uint64_t{32} << 30;
        bool verify = false;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--verify") { verify = true; continue; }
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            const std::string value = argv[++i];
            if (key == "--index-prefix") index_prefix = value;
            else if (key == "--ids-file") ids_path = value;
            else if (key == "--csv") csv_path = value;
            else if (key == "--max-nodes") maximum = std::stoull(value);
            else if (key == "--limit-bytes") limit = std::stoull(value);
            else throw std::runtime_error("Unknown option: " + key);
        }
        if (ids_path.empty() || csv_path.empty() || !maximum || maximum > UINT32_MAX || !limit)
            throw std::runtime_error("Usage: test_cache_lookup --index-prefix PREFIX --ids-file IDS.bin "
                "--max-nodes N --csv CURVE.csv --limit-bytes BYTES [--verify]");
        if (verify && maximum > 1000000)
            throw std::runtime_error("--verify is restricted to <=1M nodes to avoid duplicate large allocations");
        if (!std::filesystem::exists(ids_path)) {
            if (index_prefix.empty()) throw std::runtime_error("Index prefix required to generate IDs");
            std::vector<uint32_t> ids;
            {
                // Use production BFS. Full PQ is loaded by load(), then released BEFORE lookup insertion.
                std::shared_ptr<AlignedFileReader> reader;
#ifdef _WINDOWS
                reader = std::make_shared<WindowsAlignedFileReader>();
#else
                reader = std::make_shared<LinuxAlignedFileReader>();
#endif
                diskann::PQFlashIndex<float> index(reader, diskann::Metric::L2);
                if (index.load(1, index_prefix.c_str()) != 0) throw std::runtime_error("Cannot load disk index");
                if (maximum > static_cast<uint64_t>(std::round(index.get_num_points() * 0.5)))
                    throw std::runtime_error("Requested IDs exceed production 50% cache cap");
                index.cache_bfs_levels(maximum, ids, false);
                std::sort(ids.begin(), ids.end());
                if (ids.size() != maximum) throw std::runtime_error("BFS returned fewer IDs than requested");
                diskann::write_cache_ids(ids_path, ids);
                std::cout << "SAVED_CACHE_IDS path=" << ids_path << " nodes=" << ids.size() << std::endl;
            }
        } else {
            std::cout << "REUSE_CACHE_IDS " << ids_path << std::endl;
        }
        auto ids = diskann::read_cache_ids(ids_path, maximum);
        diskann::probe_cache_lookup(ids, limit, csv_path, verify);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Lookup probe failed: " << error.what() << std::endl;
        return 1;
    }
}
