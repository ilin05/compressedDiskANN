#!/usr/bin/env bash
export OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export OMP_PROC_BIND=FALSE
unset GOMP_CPU_AFFINITY OMP_PLACES
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
/home/zmy/compressedDiskANN/build/apps/search_disk_index --data_type float --dist_fn l2 --index_path_prefix /home/zmy/compressedDiskANN/build/data/bigann/disk_index_bigann_1B_R32_L50_A1.2 --query_file /home/zmy/compressedDiskANN/build/data/bigann/bigann_query.fbin --gt_file /home/zmy/compressedDiskANN/build/data/bigann/bigann_1B_gt.bin --cache_codec dexor --num_nodes_to_cache 10000000 --lvc_chain_max 2 --lvc_build_threads 32 --cache_load_threads 1 --lvc_fallback_root_candidates 1 --lvc_state_cache_ratio 0.01 --cache_budget_include_state --lvc_decode_profile 0 --warmup_rounds 3 --defer_round_results --search_cpu 0 -K 1 -L 1 2 3 4 5 6 8 10 -T 1 -W 2 -R 50 --result_path /home/zmy/compressedDiskANN/scripts/lvc_disk_dexor_stability_results/bigann_1B_20261117T112356Z/dexor_C10000000/search --per_round_csv /home/zmy/compressedDiskANN/scripts/lvc_disk_dexor_stability_results/bigann_1B_20261117T112356Z/dexor_C10000000/rounds.csv 
