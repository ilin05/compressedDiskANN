// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <algorithm>
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " input_uint8_bin output_float_bin" << std::endl;
        exit(-1);
    }

    std::ifstream reader(argv[1], std::ios::binary);
    if (!reader.is_open())
    {
        std::cerr << "Failed to open input file" << std::endl;
        exit(-1);
    }

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));

    size_t npts = (size_t)npts_i32;
    size_t nd = (size_t)dim_i32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << nd << std::endl;

    std::ofstream writer(argv[2], std::ios::binary);
    if (!writer.is_open())
    {
        std::cerr << "Failed to open output file" << std::endl;
        exit(-1);
    }

    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&dim_i32, sizeof(int));

    size_t block_size = 1000000;
    uint8_t *input_buf = new uint8_t[block_size * nd];
    float *output_buf = new float[block_size * nd];

    size_t pts_processed = 0;
    while (pts_processed < npts)
    {
        size_t current_block_size = std::min(block_size, npts - pts_processed);
        reader.read((char *)input_buf, current_block_size * nd * sizeof(uint8_t));
        diskann::convert_types<uint8_t, float>(input_buf, output_buf, current_block_size, nd);
        writer.write((char *)output_buf, current_block_size * nd * sizeof(float));
        pts_processed += current_block_size;
        std::cout << "Processed " << pts_processed << " / " << npts << " points." << std::endl;
    }

    delete[] output_buf;
    delete[] input_buf;

    reader.close();
    writer.close();
    
    std::cout << "Finished writing block_uint8_to_float." << std::endl;
    return 0;
}
