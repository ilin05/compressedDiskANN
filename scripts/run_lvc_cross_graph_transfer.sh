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
for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    dataset = dataset_dir.name
    for row in csv.DictReader((dataset_dir / 'hnsw_build.csv').open()):
        compression.append(dict(dataset=dataset, graph='HNSW', codec=row['Algorithm'].lower(),
            chain_max=row['ChainMaxLength'], roots=row['Roots'], max_depth=row['MaxDepth'],
            mean_reference_l2=row['MeanReferenceL2'], payload_bytes=row['PayloadBytes'],
            patch_bytes=row['F32PatchBytes'], raw_f32_bytes=row['RawF32Bytes'],
            payload_ratio_f32=row['PayloadRatioF32'], full_index_ratio_f32=row['FullIndexRatioF32']))
    for chain, suffix in ((2, ''), (4, '_ch4'), (8, '_ch8'), (-1, '_ch-1')):
        path = dataset_dir / f'vamana_lvc{suffix}_compression.csv'
        for row in csv.DictReader(path.open()):
            compression.append(dict(dataset=dataset, graph='Vamana', codec=row['codec'],
                chain_max=chain, roots=row['roots'], max_depth=row['max_depth'],
                mean_reference_l2=row['mean_reference_l2'], payload_bytes=row['payload_bytes'],
                patch_bytes=row['patch_bytes'], raw_f32_bytes=row['raw_f32_bytes'],
                payload_ratio_f32=row['payload_ratio_f32'], full_index_ratio_f32=row['full_index_ratio_f32']))
    for row in csv.DictReader((dataset_dir / 'pilot_search_summary.csv').open()):
        retrieval.append(dict(dataset=dataset, **row))
for filename, rows in (('compression_summary.csv', compression), ('retrieval_summary.csv', retrieval)):
    with (root / filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(root / filename, 'rows=', len(rows))
PY
