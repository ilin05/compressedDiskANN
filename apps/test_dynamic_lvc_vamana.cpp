// Serial repeated updates of existing Vamana graphs. The same source also
// builds against test_bigann without DISKANN_LVC_DYNAMIC for a native raw run.
#include <filesystem>
#include <numeric>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include "index.h"
#include "index_config.h"
#ifdef DISKANN_LVC_DYNAMIC
#include "dynamic_alp_data_store.h"
#include "raw_f32_data_store.h"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Store = diskann::AbstractDataStore<float>;
static double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

struct BinVectors
{
    int fd = -1;
    void *mapping = MAP_FAILED;
    size_t bytes = 0;
    uint32_t count = 0, dim = 0;
    explicit BinVectors(const std::string &path)
    {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open " + path);
        struct stat st{};
        if (fstat(fd, &st) || st.st_size < 8) throw std::runtime_error("Invalid fbin " + path);
        bytes = size_t(st.st_size);
        mapping = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("Cannot mmap " + path);
        std::memcpy(&count, mapping, 4); std::memcpy(&dim, static_cast<const char *>(mapping) + 4, 4);
        if (!count || !dim || bytes != 8 + uint64_t(count) * dim * sizeof(float))
            throw std::runtime_error("Expected dense F32 bin: " + path);
    }
    ~BinVectors() { if (mapping != MAP_FAILED) munmap(mapping, bytes); if (fd >= 0) close(fd); }
    const float *row(size_t i) const
    {
        if (i >= count) throw std::out_of_range("Input vector ID");
        return reinterpret_cast<const float *>(static_cast<const char *>(mapping) + 8) + i * dim;
    }
};

// Import a static graph without rewriting its original edges. A hidden frozen
// clone points to the old entry node; external tags are stable source IDs + 1.
namespace diskann
{
class DynamicVamanaExperiment : public diskann::Index<float, uint32_t>
{
  public:
    using diskann::Index<float, uint32_t>::Index;
    void import_static(const std::string &graph, const std::string &data, size_t expected)
    {
        if (_pq_dist || !_enable_tags || _num_frozen_pts != 1) throw std::runtime_error("Invalid import config");
        size_t rows, dimensions;
        diskann::get_bin_metadata(data, rows, dimensions);
        if (rows != expected || dimensions != _data_store->get_dims())
            throw std::runtime_error("Payload header does not match the input dataset");
        std::ifstream header(graph, std::ios::binary);
        header.exceptions(std::ios::failbit | std::ios::badbit);
        uint64_t graph_length, frozen_count; uint32_t degree, initial_entry;
        header.read(reinterpret_cast<char *>(&graph_length), 8);
        header.read(reinterpret_cast<char *>(&degree), 4);
        header.read(reinterpret_cast<char *>(&initial_entry), 4);
        header.read(reinterpret_cast<char *>(&frozen_count), 8);
        if (graph_length != fs::file_size(graph) || graph_length < 24 || frozen_count != 0 || initial_entry >= rows)
            throw std::runtime_error("Expected a complete static Vamana graph with no frozen points");
        const auto n = _data_store->load(data);
        const auto meta = _graph_store->load(graph, n);
        if (n != expected || std::get<0>(meta) != n || std::get<2>(meta) != 0 || n >= _max_points)
            throw std::runtime_error("Import requires a static graph, matching payload rows and spare capacity");
        const uint32_t entry = std::get<1>(meta);
        if (entry >= n) throw std::runtime_error("Invalid static entry point");
        _nd = n; _start = uint32_t(_max_points); _frozen_pts_used = 1;
        _data_store->copy_vectors(entry, _start, 1);
        std::vector<uint32_t> edge{entry}; _graph_store->set_neighbours(_start, edge);
        _tag_to_location.clear(); _location_to_tag.clear(); _delete_set->clear();
        // Materialize the preallocated directory during load, not the first insertion.
        _location_to_tag.set(_start, 0); _location_to_tag.erase(_start);
        for (uint32_t i = 0; i < n; ++i) { _tag_to_location[i + 1] = i; _location_to_tag.set(i, i + 1); }
        _empty_slots.clear(); _empty_slots.reserve(_max_points - n);
        for (size_t i = n; i < _max_points; ++i) _empty_slots.insert(uint32_t(i));
        _dynamic_index = true; _deletes_enabled = true; _data_compacted = false; _has_built = true;
    }
    uint32_t location(uint32_t tag) const
    {
        const auto it = _tag_to_location.find(tag);
        if (it == _tag_to_location.end()) throw std::runtime_error("Missing active tag");
        return it->second;
    }
    bool live(uint32_t tag) const { return _tag_to_location.find(tag) != _tag_to_location.end(); }
    uint32_t frozen() const { return _start; }
    size_t active() const { return _location_to_tag.size(); }
    size_t deleted() const { return _delete_set->size(); }
    uint64_t graph_bytes(bool capacity) const
    {
        uint64_t bytes = capacity ? (_max_points + 1) * sizeof(std::vector<uint32_t>) : 24;
        for (size_t i = 0; i <= _max_points; ++i)
        {
            const auto &edges = _graph_store->get_neighbours(uint32_t(i));
            if (capacity) bytes += edges.capacity() * sizeof(uint32_t);
            else if (_location_to_tag.contains(uint32_t(i)) || _delete_set->count(uint32_t(i)) || i == _start)
                bytes += 4 + edges.size() * sizeof(uint32_t);
        }
        return bytes;
    }
    uint64_t graph_hash() const
    {
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i <= _max_points; ++i)
        {
            const auto &edges = _graph_store->get_neighbours(uint32_t(i));
            h = (h ^ uint32_t(i)) * 1099511628211ULL;
            for (auto id : edges) h = (h ^ id) * 1099511628211ULL;
        }
        return h;
    }
};
} // namespace diskann

static std::vector<double> numbers(const std::string &text)
{
    std::vector<double> result; std::stringstream in(text); std::string s;
    while (std::getline(in, s, ',')) result.push_back(std::stod(s));
    if (result.empty()) throw std::runtime_error("Empty numeric list");
    return result;
}
static uint64_t rss_kib()
{
    size_t total, resident; std::ifstream in("/proc/self/statm");
    if (!(in >> total >> resident)) return 0;
    return resident * size_t(sysconf(_SC_PAGESIZE)) / 1024;
}
static uint64_t peak_kib() { struct rusage u{}; getrusage(RUSAGE_SELF, &u); return uint64_t(u.ru_maxrss); }

int main(int argc, char **argv)
{
    const auto program_start = Clock::now();
    try
    {
        std::unordered_map<std::string, std::string> opts;
        for (int i = 1; i < argc; i += 2)
        {
            if (std::string(argv[i]) == "--help")
            {
                std::cout << "--backend raw|alp --graph-prefix PATH --payload-prefix PATH --data F32BIN "
                             "--query F32BIN --gt GTBIN --output DIR [--batches 20 --ratio 0.01 --L 5,10,20 "
                             "--rounds 3 --search-every 2 --insert-alpha 0.1 --query-alpha 1,0.2 "
                             "--pq-prefix PATH --pq-bytes 32 --insert-cache 1 --dump-results 0]\n";
                return 0;
            }
            if (i + 1 == argc || std::string(argv[i]).rfind("--", 0) != 0)
                throw std::runtime_error("Options require --name value");
            if (!opts.emplace(std::string(argv[i]).substr(2), argv[i + 1]).second)
                throw std::runtime_error("Duplicate option");
        }
        const std::unordered_set<std::string> known{"backend","graph-prefix","payload-prefix","data","query","gt",
            "output","batches","ratio","L","rounds","search-every","insert-alpha","query-alpha","pq-prefix",
            "pq-bytes","insert-cache","dump-results","query-count","seed","build-L","degree","graph-alpha"};
        for (const auto &kv : opts) if (!known.count(kv.first)) throw std::runtime_error("Unknown option: " + kv.first);
        auto value = [&](const std::string &key, const std::string &def = "") {
            auto it = opts.find(key); if (it != opts.end()) return it->second;
            if (def.empty()) throw std::runtime_error("Required --" + key); return def;
        };
        const auto backend = value("backend"), outdir = value("output");
        const bool alp = backend == "alp";
        if (backend != "raw" && !alp) throw std::runtime_error("Backend must be raw or alp");
#ifndef DISKANN_LVC_DYNAMIC
        if (alp) throw std::runtime_error("This binary is the test_bigann raw baseline");
#endif
        const size_t batches = std::stoull(value("batches", "20")), rounds = std::stoull(value("rounds", "3"));
        const size_t search_every = std::stoull(value("search-every", "2"));
        const double ratio = std::stod(value("ratio", "0.01"));
        const float insert_alpha = std::stof(value("insert-alpha", alp ? "0.1" : "1"));
        const bool cache = std::stoi(value("insert-cache", "1")) != 0;
        const bool dump = std::stoi(value("dump-results", "0")) != 0;
        const auto ls = numbers(value("L", "5,10,20"));
        const auto alphas = numbers(value("query-alpha", alp ? "1,0.2" : "1"));
        if (!batches || !rounds || !search_every || !std::isfinite(ratio) || ratio <= 0 || ratio >= 1 ||
            !std::isfinite(insert_alpha) || insert_alpha <= 0 || insert_alpha > 1)
            throw std::runtime_error("Invalid workload parameters");
        for (auto l : ls) if (l < 1 || l > UINT32_MAX || l != std::floor(l)) throw std::runtime_error("Invalid L");
        for (auto a : alphas) if (a <= 0 || a > 1 || !std::isfinite(a) || (!alp && a != 1))
            throw std::runtime_error("Invalid query alpha (raw baseline is exact)");
        if (!alp && insert_alpha != 1) throw std::runtime_error("Raw baseline uses exact insertion");
        BinVectors data(value("data")), queries(value("query"));
        if (data.dim != queries.dim) throw std::runtime_error("Query dimension mismatch");
        const size_t query_limit = std::stoull(value("query-count", "0"));
        const size_t nq = query_limit ? std::min<size_t>(query_limit, queries.count) : queries.count;
        const size_t batch_size = std::max<size_t>(1, size_t(data.count * ratio));
        if (data.count >= UINT32_MAX - 1 || batches > (uint64_t(UINT32_MAX) - data.count - 2) / batch_size)
            throw std::runtime_error("Physical ID capacity overflow");
        const uint64_t cap64 = uint64_t(data.count) + uint64_t(batches) * batch_size;
        if (cap64 + 1 >= UINT32_MAX) throw std::runtime_error("Physical ID capacity overflow");
        const uint32_t capacity = uint32_t(cap64);
        std::ifstream gt(value("gt"), std::ios::binary); gt.exceptions(std::ios::badbit | std::ios::failbit);
        uint32_t gt_n, gt_k; gt.read(reinterpret_cast<char *>(&gt_n), 4); gt.read(reinterpret_cast<char *>(&gt_k), 4);
        if (gt_n < nq || !gt_k) throw std::runtime_error("GT shape mismatch");
        std::vector<uint32_t> truth(nq), gtrow(gt_k);
        for (size_t i = 0; i < nq; ++i)
        {
            gt.read(reinterpret_cast<char *>(gtrow.data()), gt_k * sizeof(uint32_t)); truth[i] = gtrow[0];
            if (truth[i] >= data.count) throw std::runtime_error("GT ID out of range");
        }
        if (fs::exists(fs::path(outdir) / "updates.csv") || fs::exists(fs::path(outdir) / "COMPLETE"))
            throw std::runtime_error("Refusing to overwrite an existing experiment; use a fresh output directory");
        fs::create_directories(outdir);
        auto wp = diskann::IndexWriteParametersBuilder(uint32_t(std::stoul(value("build-L", "50"))),
                                                       uint32_t(std::stoul(value("degree", "32"))))
                      .with_alpha(std::stof(value("graph-alpha", "1.2"))).with_num_threads(1).build();
        auto sp = diskann::IndexSearchParams(uint32_t(*std::max_element(ls.begin(), ls.end())), 1);
        auto config = diskann::IndexConfigBuilder().with_metric(diskann::L2).with_dimension(data.dim)
                          .with_max_points(capacity).with_num_frozen_pts(1).is_dynamic_index(false).is_enable_tags(true)
                          .with_index_write_params(wp).with_index_search_params(sp).with_data_type("float")
                          .with_tag_type("uint32").with_label_type("uint32").build();
        const auto load_start = Clock::now();
        std::cout << "LOAD_BEGIN backend=" << backend << " initial_nodes=" << data.count
                  << " physical_capacity=" << capacity << " update_threads=1 search_threads=1" << std::endl;
        std::shared_ptr<Store> store;
#ifdef DISKANN_LVC_DYNAMIC
        std::shared_ptr<diskann::DynamicALPDataStore> compressed;
        if (alp)
        {
            compressed = std::make_shared<diskann::DynamicALPDataStore>(capacity + 1, data.dim,
                std::unique_ptr<diskann::Distance<float>>(diskann::get_distance_function<float>(diskann::L2)));
            compressed->set_insert_cache(cache); store = compressed;
        }
        else store = std::make_shared<diskann::RawF32DataStore<float>>(capacity + 1, data.dim,
            std::unique_ptr<diskann::Distance<float>>(diskann::get_distance_function<float>(diskann::L2)));
#else
        store = std::make_shared<diskann::InMemDataStore<float>>(capacity + 1, data.dim,
            std::unique_ptr<diskann::Distance<float>>(diskann::get_distance_function<float>(diskann::L2)));
        (void)cache;
#endif
        diskann::DynamicVamanaExperiment index(config, store, std::make_unique<diskann::InMemGraphStore>(capacity + 1, 0), store);
        index.import_static(value("graph-prefix"), value("payload-prefix") + ".data", data.count);
#ifdef DISKANN_LVC_DYNAMIC
        std::shared_ptr<diskann::DynamicVamanaPQ> pq;
        const bool need_pq = insert_alpha < 1 || std::any_of(alphas.begin(), alphas.end(), [](double a) { return a < 1; });
        if (alp && (need_pq || opts.count("pq-prefix")))
        {
            pq = std::make_shared<diskann::DynamicVamanaPQ>(value("pq-prefix"), std::stoull(value("pq-bytes", "32")),
                                                           data.dim, capacity + 1, data.count);
            std::vector<float> frozen(store->get_aligned_dim(), 0);
            store->get_vector(index.frozen(), frozen.data()); pq->set_vector(index.frozen(), frozen.data());
            index.set_dynamic_pq(pq, insert_alpha, 1);
        }
#endif
        const double load_s = elapsed(load_start);
        std::cout << "LOAD_COMPLETE seconds=" << load_s << std::endl;
        std::ofstream updates(fs::path(outdir) / "updates.csv"), search(fs::path(outdir) / "search.csv"), details;
        std::ofstream timing(fs::path(outdir) / "timing.csv");
        timing.exceptions(std::ios::failbit | std::ios::badbit);
        timing << std::setprecision(12) << "stage,batch,query_alpha,L,round,seconds\n";
        auto record_time = [&](const char *stage, size_t batch, double seconds, double alpha = 0, uint32_t l = 0, size_t round = 0) {
            timing << stage << ',' << batch << ',' << alpha << ',' << l << ',' << round << ',' << seconds << '\n';
            timing.flush();
        };
        record_time("load", 0, load_s);
        updates.exceptions(std::ios::failbit | std::ios::badbit); search.exceptions(std::ios::failbit | std::ios::badbit);
        updates << std::setprecision(12); search << std::setprecision(12);
        updates << "backend,batch,active_nodes,physical_nodes,deleted_nodes,inserted_nodes,update_hash,prepare_seconds,"
            "delete_seconds,insert_seconds,insert_mean_us,delete_mean_us,payload_encode_seconds,pq_encode_seconds,"
            "graph_and_metadata_seconds,decode_calls,decode_cache_hits,decode_cache_peak_nodes,payload_bytes,"
            "payload_capacity_bytes,offset_capacity_bytes,graph_bytes,graph_capacity_bytes,pq_capacity_bytes,"
            "tag_serialized_bytes,index_accounted_bytes,current_rss_kib,peak_rss_kib,load_seconds,"
            "verified_vectors,max_abs_error,graph_hash,insert_exact_distances,insert_pq_distances\n";
        search << "backend,batch,query_alpha,L,round,queries,recall_at_1,qps,mean_query_us,empty_results\n";
        if (dump)
        {
            details.open(fs::path(outdir) / "results.csv"); details.exceptions(std::ios::failbit | std::ios::badbit);
            details << "batch,query_alpha,L,round,query,source_id,distance_bits\n";
        }
        std::mt19937_64 random(std::stoull(value("seed", "20261015")));
        std::vector<float> decoded(store->get_aligned_dim(), 0);
        for (size_t batch = 0; batch <= batches; ++batch)
        {
            const auto prep_start = Clock::now();
            std::vector<uint32_t> ids; std::unordered_set<uint32_t> selected;
            if (batch)
                for (uint64_t j = data.count - batch_size; j < data.count; ++j)
                {
                    const auto t = std::uniform_int_distribution<uint32_t>(0, uint32_t(j))(random);
                    const auto id = selected.count(t) ? uint32_t(j) : t;
                    selected.insert(id); ids.push_back(id);
                }
            std::vector<float> input(ids.size() * data.dim);
            for (size_t i = 0; i < ids.size(); ++i) std::memcpy(input.data() + i * data.dim, data.row(ids[i]), data.dim * 4);
            const double prepare_s = elapsed(prep_start);
            record_time("prepare", batch, prepare_s);
            uint64_t hash = 1469598103934665603ULL;
            for (auto id : ids) hash = (hash ^ id) * 1099511628211ULL;
            double delete_s = 0, insert_s = 0, enc_before = 0, pq_before = 0;
            uint64_t dec_before = 0, hits_before = 0, exact_before = 0, pq_dist_before = 0;
#ifdef DISKANN_LVC_DYNAMIC
            if (compressed) { enc_before = compressed->encode_seconds; dec_before = compressed->payload_decodes; hits_before = compressed->cache_hits; }
            pq_before = index.dynamic_pq_encode_seconds();
            if (compressed) exact_before = compressed->insert_distance_evaluations;
            if (pq) pq_dist_before = pq->distance_evaluations;
            index.set_dynamic_query_alpha(1);
#endif
            if (batch)
            {
                auto start = Clock::now();
                for (auto id : ids) if (index.lazy_delete(id + 1)) throw std::runtime_error("Lazy deletion failed");
                delete_s = elapsed(start);
                for (auto id : ids) if (index.live(id + 1)) throw std::runtime_error("Deleted tag still active");
                // Check logical deletion through search before reinsertion, outside the update timers.
                for (size_t i = 0; i < std::min<size_t>(16, ids.size()); ++i)
                {
                    uint32_t tag = 0; std::vector<float *> vectors;
                    const auto count = index.search_with_tags(input.data() + i * data.dim, 1,
                        uint32_t(*std::max_element(ls.begin(), ls.end())), &tag, nullptr, vectors);
                    if (count && (!tag || selected.count(tag - 1)))
                        throw std::runtime_error("Logically deleted vector returned by search");
                }
                start = Clock::now();
                for (size_t i = 0; i < ids.size(); ++i)
                {
                    if (index.insert_point(input.data() + i * data.dim, ids[i] + 1)) throw std::runtime_error("Insertion failed");
                    if ((i + 1) % 100000 == 0)
                        std::cout << "UPDATE_PROGRESS batch=" << batch << " inserted=" << i + 1
                                  << '/' << ids.size() << " seconds=" << elapsed(start) << std::endl;
                }
                insert_s = elapsed(start);
            }
            record_time("delete", batch, delete_s); record_time("insert", batch, insert_s);
            double enc_s = 0, pq_s = 0; uint64_t decodes = 0, hits = 0, cache_peak = 0, exact_distances = 0, pq_distances = 0;
#ifdef DISKANN_LVC_DYNAMIC
            if (compressed)
            {
                enc_s = compressed->encode_seconds - enc_before; decodes = compressed->payload_decodes - dec_before;
                hits = compressed->cache_hits - hits_before; cache_peak = compressed->cache_peak_nodes;
                exact_distances = compressed->insert_distance_evaluations - exact_before;
            }
            if (pq) pq_distances = pq->distance_evaluations - pq_dist_before;
            pq_s = index.dynamic_pq_encode_seconds() - pq_before;
#else
            (void)enc_before; (void)pq_before; (void)dec_before; (void)hits_before; (void)exact_before; (void)pq_dist_before;
#endif
            if (index.active() != data.count || index.deleted() != batch * batch_size ||
                index.get_num_points() != data.count + batch * batch_size)
                throw std::runtime_error("Active/physical/deleted count mismatch");
            const auto verify_start = Clock::now();
            double max_error = 0; size_t verified = 0;
            auto verify = [&](uint32_t id) {
                uint32_t tag = id + 1;
                if (index.get_vector_by_tag(tag, decoded.data())) throw std::runtime_error("Missing inserted payload");
                for (size_t d = 0; d < data.dim; ++d)
                {
                    double e = std::abs(double(decoded[d]) - data.row(id)[d]);
                    if (!std::isfinite(e) || e > (alp ? 1.1e-4 : 0)) throw std::runtime_error("Payload reconstruction mismatch");
                    max_error = std::max(max_error, e);
                }
#ifdef DISKANN_LVC_DYNAMIC
                if (pq && batch && selected.count(id) && !pq->verify(index.location(tag), data.row(id)))
                    throw std::runtime_error("Inserted PQ code mismatch");
#endif
                ++verified;
            };
            for (auto id : ids) verify(id);
            for (size_t j = 0; j < std::min<size_t>(1024, data.count); ++j) verify(uint32_t(j * uint64_t(data.count) / std::min<size_t>(1024, data.count)));
            record_time("verify", batch, elapsed(verify_start));
            const auto stats_start = Clock::now();
            uint64_t payload = uint64_t(index.get_num_points() + 1) * data.dim * 4;
            uint64_t payload_cap = uint64_t(capacity + 1) * store->get_aligned_dim() * 4, offsets = 0, pq_bytes = 0;
#ifdef DISKANN_LVC_DYNAMIC
            if (compressed) { payload = compressed->payload_bytes; payload_cap = compressed->payload_capacity_bytes(); offsets = compressed->metadata_bytes(); }
            if (pq) pq_bytes = pq->bytes();
#endif
            const auto graph = index.graph_bytes(false), graph_cap = index.graph_bytes(true);
            const uint64_t tag_bytes = uint64_t(index.get_num_points() + 1 + index.deleted()) * 4;
            updates << backend << ',' << batch << ',' << index.active() << ',' << index.get_num_points() << ','
                << index.deleted() << ',' << ids.size() << ',' << hash << ',' << prepare_s << ',' << delete_s << ','
                << insert_s << ',' << (ids.empty() ? 0 : insert_s * 1e6 / ids.size()) << ','
                << (ids.empty() ? 0 : delete_s * 1e6 / ids.size()) << ',' << enc_s << ',' << pq_s << ','
                << std::max(0.0, insert_s - enc_s - pq_s) << ',' << decodes << ',' << hits << ',' << cache_peak << ','
                << payload << ',' << payload_cap << ',' << offsets << ',' << graph << ',' << graph_cap << ',' << pq_bytes << ','
                << tag_bytes << ',' << payload + offsets + graph + pq_bytes + tag_bytes << ',' << rss_kib() << ','
                << peak_kib() << ',' << load_s << ',' << verified << ',' << max_error << ',' << (dump ? index.graph_hash() : 0)
                << ',' << exact_distances << ',' << pq_distances << '\n';
            updates.flush();
            record_time("storage_accounting", batch, elapsed(stats_start));
            std::cout << "PASS update backend=" << backend << " batch=" << batch << " insert_us="
                      << (ids.empty() ? 0 : insert_s * 1e6 / ids.size()) << " max_error=" << max_error << std::endl;
            if (batch % search_every && batch != batches) continue;
            for (double alpha : alphas)
            {
#ifdef DISKANN_LVC_DYNAMIC
                index.set_dynamic_query_alpha(float(alpha));
#endif
                for (double l : ls)
                {
                    std::vector<float *> vectors;
                    const auto warmup_start = Clock::now();
                    for (size_t q = 0; q < std::min<size_t>(100, nq); ++q)
                    { uint32_t tag = UINT32_MAX; index.search_with_tags(queries.row(q), 1, uint32_t(l), &tag, nullptr, vectors); }
                    record_time("warmup", batch, elapsed(warmup_start), alpha, uint32_t(l));
                    for (size_t round = 0; round < rounds; ++round)
                    {
                        std::vector<uint32_t> tags(nq, UINT32_MAX); std::vector<float> distances(nq, 0);
                        std::vector<size_t> counts(nq);
                        const auto start = Clock::now();
                        for (size_t q = 0; q < nq; ++q)
                            counts[q] = index.search_with_tags(queries.row(q), 1, uint32_t(l), &tags[q], &distances[q], vectors);
                        const double secs = elapsed(start); size_t matches = 0, empty = 0;
                        record_time("query", batch, secs, alpha, uint32_t(l), round);
                        for (size_t q = 0; q < nq; ++q)
                        {
                            uint32_t source = UINT32_MAX, bits = 0;
                            if (!counts[q]) ++empty;
                            else
                            {
                                if (!tags[q] || tags[q] > data.count || !index.live(tags[q]))
                                    throw std::runtime_error("Deleted/invalid/frozen result returned");
                                source = tags[q] - 1; matches += source == truth[q];
                                std::memcpy(&bits, &distances[q], 4);
                            }
                            if (dump) details << batch << ',' << alpha << ',' << uint32_t(l) << ',' << round << ','
                                              << q << ',' << source << ',' << bits << '\n';
                        }
                        search << backend << ',' << batch << ',' << alpha << ',' << uint32_t(l) << ',' << round << ','
                               << nq << ',' << double(matches) / nq << ',' << nq / secs << ',' << secs * 1e6 / nq << ',' << empty << '\n';
                        search.flush(); if (dump) details.flush();
                    }
                }
            }
        }
        record_time("total_wall", batches, elapsed(program_start));
        std::ofstream complete(fs::path(outdir) / "COMPLETE");
        complete.exceptions(std::ios::failbit | std::ios::badbit);
        complete << "PASS\n"; complete.close();
        return 0;
    }
    catch (const std::exception &e) { std::cerr << "FAIL dynamic Vamana: " << e.what() << std::endl; return 1; }
}
