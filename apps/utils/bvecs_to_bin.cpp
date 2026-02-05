// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <cstring>
#include "utils.h"

// Convert block of bvecs to bin
void block_convert_bvecs_to_bin(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf, size_t npts, size_t ndims)
{
    // bvecs format: [dim (4 bytes)] [vector (dim bytes)]
    // We read npts * (ndims + 4) bytes
    reader.read((char *)read_buf, npts * (ndims + sizeof(uint32_t)));

    for (size_t i = 0; i < npts; i++)
    {
        uint32_t dim_check;
        memcpy(&dim_check, read_buf + i * (ndims + sizeof(uint32_t)), sizeof(uint32_t));
        
        if (dim_check != ndims) {
            std::cerr << "Error: Dimension mismatch at point " << i << ". Expected " << ndims << ", got " << dim_check << std::endl;
            exit(-1);
        }

        // Copy vector data (skip the 4-byte dimension header)
        memcpy(write_buf + i * ndims, read_buf + i * (ndims + sizeof(uint32_t)) + sizeof(uint32_t), ndims);
    }
    
    // Write just the raw vector data
    writer.write((char *)write_buf, npts * ndims);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " input_bvecs output_bin" << std::endl;
        exit(-1);
    }

    std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
    if (!reader.is_open()) {
        std::cerr << "Error: Could not open input file " << argv[1] << std::endl;
        exit(-1);
    }

    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);
    size_t ndims = (size_t)ndims_u32;
    
    // bvecs size per point: 4 bytes (dim) + ndims bytes (data)
    size_t point_size = ndims + sizeof(uint32_t);
    size_t npts = fsize / point_size;
    
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    size_t blk_size = 131072; // Process in chunks
    size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;

    std::ofstream writer(argv[2], std::ios::binary);
    if (!writer.is_open()) {
        std::cerr << "Error: Could not open output file " << argv[2] << std::endl;
        exit(-1);
    }

    // DiskANN bin header: num_points (int32), num_dims (int32)
    int32_t npts_s32 = (int32_t)npts;
    int32_t ndims_s32 = (int32_t)ndims;
    writer.write((char *)&npts_s32, sizeof(int32_t));
    writer.write((char *)&ndims_s32, sizeof(int32_t));

    size_t chunknpts = std::min(npts, blk_size);
    uint8_t *read_buf = new uint8_t[chunknpts * point_size];
    uint8_t *write_buf = new uint8_t[chunknpts * ndims];

    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert_bvecs_to_bin(reader, writer, read_buf, write_buf, cblk_size, ndims);
        if (i % 100 == 0) { // Print progress periodically
            std::cout << "Block #" << i << " / " << nblks << " written" << std::endl;
        }
    }
    std::cout << "Conversion completed successfully." << std::endl;

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();

    return 0;
}
