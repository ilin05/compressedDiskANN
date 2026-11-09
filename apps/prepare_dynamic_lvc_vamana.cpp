// Prepare a shared F32 graph and the legacy ALP payload file for correctness runs.
#include "index.h"
#include "raw_f32_data_store.h"
#include <filesystem>
int main(int argc, char **argv)
{
    try
    {
        if (argc != 7) throw std::runtime_error("Usage: prepare_dynamic_lvc_vamana DATA_FBIN RAW_PREFIX ALP_PREFIX R L THREADS");
        uint32_t n, d; std::ifstream in(argv[1], std::ios::binary); in.exceptions(std::ios::badbit | std::ios::failbit);
        in.read(reinterpret_cast<char *>(&n),4); in.read(reinterpret_cast<char *>(&d),4);
        if (!n || !d) throw std::runtime_error("Empty input");
        auto wp = diskann::IndexWriteParametersBuilder(std::stoul(argv[5]),std::stoul(argv[4]))
            .with_alpha(1.2f).with_num_threads(std::stoul(argv[6])).build();
        auto config = diskann::IndexConfigBuilder().with_metric(diskann::L2).with_dimension(d).with_max_points(n)
            .with_num_frozen_pts(0).is_dynamic_index(false).is_enable_tags(false).with_index_write_params(wp)
            .with_data_type("float").with_tag_type("uint32").with_label_type("uint32").build();
        if (std::filesystem::exists(argv[2]) != std::filesystem::exists(std::string(argv[2])+".data"))
            throw std::runtime_error("Incomplete raw index: graph and .data must both exist or both be absent");
        if (!std::filesystem::exists(argv[2]))
        {
            auto store = std::make_shared<diskann::RawF32DataStore<float>>(n,d,
                std::unique_ptr<diskann::Distance<float>>(diskann::get_distance_function<float>(diskann::L2)));
            diskann::Index<float,uint32_t> index(config,store,std::make_unique<diskann::InMemGraphStore>(n,0),store);
            index.build(argv[1],n); index.save(argv[2]);
        }
        // Use the existing codec's own encoder and serialization, not a Python reimplementation.
        if (!std::filesystem::exists(std::string(argv[3])+".data"))
        {
            diskann::InMemDataStore<float> alp(n,d,
                std::unique_ptr<diskann::Distance<float>>(diskann::get_distance_function<float>(diskann::L2)));
            alp.populate_data(argv[1],0); alp.save(std::string(argv[3])+".data",n);
        }
        std::cout << "PASS prepared shared static graph and legacy ALP payloads" << std::endl;
        return 0;
    }
    catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
