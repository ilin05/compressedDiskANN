#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${CROSS_GRAPH_OUTPUT_ROOT:-$SCRIPT_DIR/../build/data/cross_graph_transfer}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
for dataset in \
  sift-128-euclidean \
  mnist-784-euclidean \
  fashion-mnist-784-euclidean \
  gist-960-euclidean \
  deep-image-96-angular; do
  echo "Cross-graph transfer: $dataset"
  DATASET="$dataset" OUTPUT_DIR="$OUTPUT_ROOT/$dataset" \
    bash "$SCRIPT_DIR/test_lvc_cross_graph_sift.sh"
done

python3 - "$OUTPUT_ROOT" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
compression, retrieval = [], []
datasets = ('sift-128-euclidean', 'mnist-784-euclidean',
            'fashion-mnist-784-euclidean', 'gist-960-euclidean',
            'deep-image-96-angular')
for dataset_dir in (root / name for name in datasets):
    dataset = dataset_dir.name
    hnsw_rows = list(csv.DictReader((dataset_dir / 'hnsw_build.csv').open()))
    if len(hnsw_rows) != 12:
        raise SystemExit(f'{dataset}: expected 12 HNSW codec/chain rows, got {len(hnsw_rows)}')
    for row in hnsw_rows:
        compression.append(dict(dataset=dataset, graph='HNSW', codec=row['Algorithm'].lower(),
            chain_max=row['ChainMaxLength'], target_roots=row['InitialRoots'],
            roots=row['Roots'], max_depth=row['MaxDepth'],
            average_depth=row['AverageDepth'], average_nonroot_depth=row['AverageNonrootDepth'],
            mean_reference_l2=row['MeanReferenceL2'],
            p50_reference_l2=row['P50ReferenceL2'], p95_reference_l2=row['P95ReferenceL2'],
            payload_bytes=row['PayloadBytes'],
            patch_bytes=row['F32PatchBytes'], raw_f32_bytes=row['RawF32Bytes'],
            graph_bytes=row['GraphMetadataBytes'], metadata_bytes=0,
            state_cache_bytes=row['StateCacheBytes'],
            raw_f32_index_bytes=row['RawF32IndexBytes'],
            compressed_index_bytes=row['CompressedIndexBytes'],
            payload_ratio_f32=row['PayloadRatioF32'],
            full_index_ratio_f32=row['FullIndexRatioF32'], record_file_materialized=1))
    for chain, suffix in ((2, ''), (4, '_ch4'), (8, '_ch8'), (-1, '_ch-1')):
        path = dataset_dir / f'vamana_lvc{suffix}_compression.csv'
        vamana_rows = list(csv.DictReader(path.open()))
        if len(vamana_rows) != 3:
            raise SystemExit(f'{dataset} chain={chain}: expected 3 Vamana codec rows')
        for row in vamana_rows:
            compression.append(dict(dataset=dataset, graph='Vamana', codec=row['codec'],
                chain_max=chain, target_roots=row['roots'], roots=row['roots'],
                max_depth=row['max_depth'],
                average_depth=row['average_depth'],
                average_nonroot_depth=row['average_nonroot_depth'],
                mean_reference_l2=row['mean_reference_l2'], payload_bytes=row['payload_bytes'],
                p50_reference_l2=row['p50_reference_l2'],
                p95_reference_l2=row['p95_reference_l2'],
                patch_bytes=row['patch_bytes'], raw_f32_bytes=row['raw_f32_bytes'],
                graph_bytes=row['graph_bytes'], metadata_bytes=row['metadata_bytes'],
                state_cache_bytes=0,
                raw_f32_index_bytes=row['raw_f32_index_bytes'],
                compressed_index_bytes=row['compressed_index_bytes'],
                payload_ratio_f32=row['payload_ratio_f32'],
                full_index_ratio_f32=row['full_index_ratio_f32'],
                record_file_materialized=row['record_file_materialized']))
    search_rows = list(csv.DictReader((dataset_dir / 'retrieval_ch2.csv').open()))
    if not search_rows:
        raise SystemExit(f'{dataset}: no chain=2 retrieval results')
    retrieval.extend(search_rows)
for filename, rows in (('compression_summary.csv', compression), ('retrieval_summary.csv', retrieval)):
    with (root / filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(root / filename, 'rows=', len(rows))
with (root / 'retrieval_alpha_0p2.csv').open('w', newline='') as f:
    rows = [row for row in retrieval if row['tls'] == '1' and abs(float(row['alpha']) - 0.2) < 1e-6]
    writer = csv.DictWriter(f, fieldnames=retrieval[0].keys())
    writer.writeheader(); writer.writerows(rows)
print(root / 'retrieval_alpha_0p2.csv', 'rows=', len(rows))
PY
