#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
BIGANN_DIR="$BUILD_DIR/data/bigann"
SIFT_DIR="$BUILD_DIR/data/sift"
MNIST_DIR="$BUILD_DIR/data/mnist"
GIST_DIR="$BUILD_DIR/data/gist"
FASHION_DIR="$BUILD_DIR/data/fashion-mnist"
DEEP_DIR="$BUILD_DIR/data/deep"

"$BUILD_DIR/apps/search_disk_index" \

# ../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $BIGANN_DIR/bigann_500M.fbin --index_path_prefix $BIGANN_DIR/memory_index_bigann_500M_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2
# ../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $BIGANN_DIR/bigann_base.fbin --index_path_prefix $BIGANN_DIR/memory_index_bigann_1B_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2
../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $SIFT_DIR/sift_base.fbin --index_path_prefix $SIFT_DIR/memory_index_sift_1M_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2
../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $MNIST_DIR/mnist_base.fbin --index_path_prefix $MNIST_DIR/memory_index_mnist_60K_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2
../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $GIST_DIR/gist_base.fbin --index_path_prefix $GIST_DIR/memory_index_gist_1M_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2
../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $FASHION_DIR/fashion-mnist_base.fbin --index_path_prefix $FASHION_DIR/memory_index_fashion-mnist_60K_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2
../build/apps/build_memory_index --data_type float --dist_fn l2 --data_path $DEEP_DIR/deep_base.fbin --index_path_prefix $DEEP_DIR/memory_index_deep_10M_R32_L50_A1.2 -R 32 -L 50 -T 32 --alpha 1.2