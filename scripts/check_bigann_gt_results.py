#!/usr/bin/env python3
"""Inspect F32 BIGANN GT and saved res bins without loading an index or rerunning search.

DiskANN GT layout: uint32[nq, k], all uint32 IDs, then optional float32 distances.
Result/F32 vector layout: uint32[n, d], then the dense array (no row headers).
All reported recalls are percentages. Distance checks use squared L2.
Inputs are read-only; output CSV/JSON files go to a new timestamped directory.
Requires numpy.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys

import numpy as np


def header(path):
    with path.open('rb') as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError(f'Truncated header: {path}')
    n, d = struct.unpack('<II', raw)
    if not n or not d:
        raise ValueError(f'Empty/invalid shape ({n}, {d}): {path}')
    return n, d


def matrix(path, dtype):
    n, d = header(path)
    expected = 8 + n * d * 4
    if path.stat().st_size != expected:
        raise ValueError(f'{path}: size {path.stat().st_size}, expected {expected}; '
                         'expected a dense 4-byte bin, not fvecs/ivecs')
    return np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(n, d))


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size,
                mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect_gt(ids, distances, base_count, chunk):
    n, k = ids.shape
    stats = dict(queries=n, neighbors=k, has_distances=distances is not None,
                 invalid_ids=0, all_zero_distance_rows=0, all_equal_distance_rows=0,
                 nonfinite_distances=0, negative_distances=0, unsorted_distance_rows=0)
    ties = np.ones(n, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        stats['invalid_ids'] += int(np.count_nonzero(ids[start:stop] >= base_count))
        if distances is None:
            continue
        block = np.asarray(distances[start:stop])
        finite = np.isfinite(block)
        stats['nonfinite_distances'] += int(np.count_nonzero(~finite))
        stats['negative_distances'] += int(np.count_nonzero(block < 0))
        stats['all_zero_distance_rows'] += int(np.count_nonzero(np.all(block == 0, axis=1)))
        equal = block == block[:, :1]
        stats['all_equal_distance_rows'] += int(np.count_nonzero(np.all(equal, axis=1)))
        stats['unsorted_distance_rows'] += int(np.count_nonzero(np.any(block[:, 1:] < block[:, :-1], axis=1)))
        # Match calculate_recall exactly: the consecutive equal prefix, no epsilon.
        ties[start:stop] = np.cumprod(equal, axis=1, dtype=np.int64).sum(axis=1)
    stats.update(multiple_first_ties_percent=100.0 * float(np.mean(ties > 1)),
                 first_ties_mean=float(np.mean(ties)), first_ties_median=float(np.median(ties)),
                 first_ties_max=int(np.max(ties)))
    return stats, ties


def distance_row(base, queries, query, source, stored, atol, rtol):
    row = dict(query=query, source_id=source, stored_squared_l2=stored)
    if source < 0 or source >= len(base):
        return dict(row, valid_id=False, recomputed_squared_l2='', abs_error='', distance_matches=False)
    delta = np.asarray(base[source], dtype=np.float64) - np.asarray(queries[query], dtype=np.float64)
    exact = float(np.dot(delta, delta))
    return dict(row, valid_id=True, recomputed_squared_l2=exact,
                abs_error='' if stored is None else abs(exact - stored),
                distance_matches='' if stored is None else bool(np.isclose(stored, exact, atol=atol, rtol=rtol)))


def recall(ids, ties, results, base_count, chunk):
    strict = tied = invalid = 0
    ranks = np.arange(ids.shape[1])[None, :]
    for start in range(0, len(ids), chunk):
        stop = min(start + chunk, len(ids))
        returned = np.asarray(results[start:stop, 0])
        valid = returned < base_count
        invalid += int(np.count_nonzero(~valid))
        strict += int(np.count_nonzero(valid & (returned == ids[start:stop, 0])))
        hits = (ids[start:stop] == returned[:, None]) & (ranks < ties[start:stop, None])
        tied += int(np.count_nonzero(valid & np.any(hits, axis=1)))
    return dict(strict_recall_at_1_percent=100.0 * strict / len(ids),
                stored_gt_diskann_tie_aware_recall_at_1_percent=100.0 * tied / len(ids),
                invalid_top1_result_ids=invalid)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=repo / 'build/data/bigann')
    parser.add_argument('--base', type=Path, help='F32 bin; default DATA_DIR/bigann_100M.fbin')
    parser.add_argument('--queries', type=Path, help='F32 bin; default DATA_DIR/bigann_query.fbin')
    parser.add_argument('--gt', type=Path, help='Default DATA_DIR/bigann_100M_gt.bin')
    parser.add_argument('--result-dir', type=Path, help='Default DATA_DIR')
    parser.add_argument('--result-glob', default='res_*_idx_uint32.bin')
    parser.add_argument('--query-ids', type=int, nargs='+', default=[0, 1, 2], help='Zero-based query IDs for sparse distance checks')
    parser.add_argument('--gt-ranks', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='Zero-based GT ranks for sparse distance checks')
    parser.add_argument('--chunk-queries', type=int, default=256)
    parser.add_argument('--atol', type=float, default=1e-3, help='Tolerance for recomputed distances only, never tie matching')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.chunk_queries < 1 or args.atol < 0 or args.rtol < 0:
        parser.error('chunk-queries must be positive; tolerances must be nonnegative')
    base_path = args.base or args.data_dir / 'bigann_100M.fbin'
    query_path = args.queries or args.data_dir / 'bigann_query.fbin'
    gt_path = args.gt or args.data_dir / 'bigann_100M_gt.bin'
    result_dir = args.result_dir or args.data_dir
    base = matrix(base_path, '<f4')
    queries = matrix(query_path, '<f4')
    nq, k = header(gt_path)
    if queries.shape != (nq, base.shape[1]):
        raise ValueError(f'Incompatible base/query/GT shapes: {base.shape}, {queries.shape}, {(nq, k)}')
    size = gt_path.stat().st_size
    if size not in (8 + nq * k * 4, 8 + nq * k * 8):
        raise ValueError(f'GT size {size} incompatible with shape {(nq, k)}')
    ids = np.memmap(gt_path, dtype='<u4', mode='r', offset=8, shape=(nq, k))
    distances = (np.memmap(gt_path, dtype='<f4', mode='r', offset=8 + nq * k * 4, shape=(nq, k))
                 if size == 8 + nq * k * 8 else None)
    sample_queries = list(dict.fromkeys(args.query_ids))
    sample_ranks = list(dict.fromkeys(args.gt_ranks))
    if any(q < 0 or q >= nq for q in sample_queries) or any(r < 0 or r >= k for r in sample_ranks):
        raise ValueError(f'query-ids must be in [0, {nq}); gt-ranks must be in [0, {k})')

    stats, ties = inspect_gt(ids, distances, len(base), args.chunk_queries)
    gt_samples = []
    for q in sample_queries:
        for rank in sample_ranks:
            stored = None if distances is None else float(distances[q, rank])
            gt_samples.append(dict(gt_rank=rank, **distance_row(base, queries, q, int(ids[q, rank]), stored, args.atol, args.rtol)))
    mismatch_count = sum(row['distance_matches'] is False for row in gt_samples)
    warnings = []
    if mismatch_count:
        warnings.append(f'{mismatch_count} sampled GT entries fail direct squared-L2 checks; stored-distance tie-aware recall is untrusted.')
    if distances is not None and k > 1 and stats['all_zero_distance_rows']:
        warnings.append(f"GT contains {stats['all_zero_distance_rows']} all-zero distance rows; verify that these are real duplicates before interpreting ties.")
    for field in ('invalid_ids', 'nonfinite_distances', 'negative_distances', 'unsorted_distance_rows'):
        if stats[field]:
            warnings.append(f'GT {field}={stats[field]}')
    validation = ('sample_mismatch_or_gt_anomaly' if warnings else
                  'sample_checks_passed_not_full_gt_validation' if distances is not None else 'ids_only_no_distance_validation')
    print(f'Base shape: {base.shape}; query shape: {queries.shape}; GT shape: {(nq, k)}', flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    print('Sampled GT squared-L2 checks (zero-based query/rank):', flush=True)
    for row in gt_samples:
        print(f"  q={row['query']} rank={row['gt_rank']} id={row['source_id']} stored={row['stored_squared_l2']} "
              f"recomputed={row['recomputed_squared_l2']} match={row['distance_matches']}", flush=True)

    summaries, result_samples, manifest = [], [], []
    paths = sorted(result_dir.glob(args.result_glob))
    for path in paths:
        if not path.name.endswith('_idx_uint32.bin'):
            raise ValueError(f'Expected *_idx_uint32.bin result file, got {path}')
        distance_path = path.with_name(path.name[:-len('_idx_uint32.bin')] + '_dists_float.bin')
        row = dict(result_file=str(path.resolve()), distance_file=str(distance_path.resolve()),
                   provenance='unknown_from_bin_headers', gt_validation=validation)
        entry = dict(ids=identity(path))
        if distance_path.is_file():
            entry['distances'] = identity(distance_path)
        manifest.append(entry)
        try:
            result = matrix(path, '<u4')
            if len(result) != nq:
                raise ValueError(f'Result has {len(result)} queries, GT has {nq}')
            row.update(query_count=nq, saved_K=result.shape[1], **recall(ids, ties, result, len(base), args.chunk_queries))
            result_distances = None
            if distance_path.is_file():
                try:
                    result_distances = matrix(distance_path, '<f4')
                    if result_distances.shape != result.shape:
                        raise ValueError(f'ID/distance shape mismatch: {result.shape} vs {result_distances.shape}')
                    row['distance_pair_status'] = 'shape_matches_provenance_unverified'
                except (OSError, ValueError) as error:
                    result_distances = None
                    row['distance_pair_status'] = str(error)
                    warnings.append(f'{path.name}: {error}')
            else:
                row['distance_pair_status'] = 'missing'
                warnings.append(f'Missing paired distances: {distance_path}')
            for q in sample_queries:
                stored = None if result_distances is None else float(result_distances[q, 0])
                sample = dict(result_file=path.name, saved_rank=0,
                              **distance_row(base, queries, q, int(result[q, 0]), stored, args.atol, args.rtol))
                sample['gt_first_id'] = int(ids[q, 0])
                sample['recomputed_gt_first_squared_l2'] = distance_row(base, queries, q, int(ids[q, 0]), None, args.atol, args.rtol)['recomputed_squared_l2']
                result_samples.append(sample)
            print(f"{path.name}: saved_K={row['saved_K']} strict@1={row['strict_recall_at_1_percent']:.4f}% "
                  f"stored-GT tie-aware@1={row['stored_gt_diskann_tie_aware_recall_at_1_percent']:.4f}% "
                  f"[{validation}]", flush=True)
        except (OSError, ValueError) as error:
            row['error'] = str(error)
            warnings.append(f'{path.name}: {error}')
        summaries.append(row)
    if not paths:
        warnings.append(f'No ID result files match {result_dir / args.result_glob}')
    orphan_distances = [str(p.resolve()) for p in sorted(result_dir.glob('*_dists_float.bin'))
                        if not p.with_name(p.name[:-len('_dists_float.bin')] + '_idx_uint32.bin').is_file()]
    if orphan_distances:
        warnings.append(f'{len(orphan_distances)} distance bins have no paired ID bin; see report.json.')
    warnings.append('res bin headers contain only shape, not codec/batch/round/index/query identity; matching shapes do not establish provenance.')
    warnings.append('Saved distance mismatches may indicate approximate scoring, incorrect input vectors, or mismatched/overwritten result pairs; they alone do not prove file corruption.')
    report = dict(inputs=dict(base=identity(base_path), queries=identity(query_path), gt=identity(gt_path)),
                  base_shape=list(base.shape), query_shape=list(queries.shape), gt=stats,
                  gt_sample_mismatches=mismatch_count, gt_validation=validation,
                  query_ids=sample_queries, gt_ranks=sample_ranks, atol=args.atol, rtol=args.rtol,
                  result_files=manifest, orphan_distance_files=orphan_distances, warnings=warnings,
                  recall_definition='Percent; first saved result ID; exact consecutive ties in stored GT distances, or strict matching for IDs-only GT. No GT repair performed.')
    output = args.output_dir or repo / 'scripts/bigann_gt_check_results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'gt_samples.csv', gt_samples)
    write_csv(output / 'result_samples.csv', result_samples)
    write_csv(output / 'recall_summary.csv', summaries)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for warning in warnings:
        print(f'WARNING: {warning}', flush=True)
    print(f'Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
