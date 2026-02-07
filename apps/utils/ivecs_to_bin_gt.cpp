// apps/utils/ivecs_to_bin_gt.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <idx_file.ivecs> <dis_file.fvecs> <output_gt.bin>" << std::endl;
        return 1;
    }

    std::string idx_path = argv[1];
    std::string dis_path = argv[2];
    std::string out_path = argv[3];

    std::ifstream idx_in(idx_path, std::ios::binary);
    std::ifstream dis_in(dis_path, std::ios::binary);
    std::ofstream out(out_path, std::ios::binary);

    if (!idx_in || !dis_in || !out) {
        std::cerr << "Error opening files." << std::endl;
        return 1;
    }

    // 读取第一个向量的维度(k)来推断
    int k_idx, k_dis;
    idx_in.read((char*)&k_idx, 4);
    dis_in.read((char*)&k_dis, 4);

    if (k_idx != k_dis) {
        std::cerr << "Mismatch k: " << k_idx << " vs " << k_dis << std::endl;
        return 1;
    }
    
    // 重置文件指针
    idx_in.seekg(0, std::ios::end);
    size_t idx_size = idx_in.tellg();
    idx_in.seekg(0, std::ios::beg);

    // 计算查询数量
    // ivecs 格式: 每个向量占 4 + k*4 字节
    size_t vector_size = 4 + k_idx * 4;
    size_t num_queries = idx_size / vector_size;

    std::cout << "Detected: " << num_queries << " queries, Top-K = " << k_idx << std::endl;

    // 准备数据容器
    std::vector<uint32_t> all_ids;
    std::vector<float> all_dists;
    all_ids.reserve(num_queries * k_idx);
    all_dists.reserve(num_queries * k_idx);

    for (size_t i = 0; i < num_queries; ++i) {
        int dim;
        // 读取 IDs
        idx_in.read((char*)&dim, 4);
        std::vector<int> ids(dim); // ivecs 实际上是 int32
        idx_in.read((char*)ids.data(), dim * sizeof(int));

        // 读取 Dists
        dis_in.read((char*)&dim, 4);
        std::vector<float> dists(dim);
        dis_in.read((char*)dists.data(), dim * sizeof(float));

        for (int j = 0; j < k_idx; ++j) {
            all_ids.push_back((uint32_t)ids[j]);
            all_dists.push_back(dists[j]);
        }
    }

    // 写入 DiskANN 格式 Header
    int nq = (int)num_queries;
    int k = k_idx;
    out.write((char*)&nq, 4);
    out.write((char*)&k, 4);

    // 写入 IDs 数据块
    out.write((char*)all_ids.data(), all_ids.size() * sizeof(uint32_t));
    
    // 写入 Dists 数据块
    out.write((char*)all_dists.data(), all_dists.size() * sizeof(float));

    std::cout << "Successfully converted to " << out_path << std::endl;
    return 0;
}