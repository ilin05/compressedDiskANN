#!/usr/bin/env python3
"""Summarize completed HNSW/Vamana cross-graph transfer datasets."""

import argparse
import csv
from pathlib import Path


CHAINS = ((2, ""), (4, "_ch4"), (8, "_ch8"), (-1, "_ch-1"))
CODECS = ("dexor", "gorilla", "elf")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "lvc_cross_graph_transfer_results"
DATASET_ORDER = (
    "sift-128-euclidean", "mnist-784-euclidean",
    "fashion-mnist-784-euclidean", "gist-960-euclidean",
    "deep-image-96-angular",
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    print(f"{path}: {len(rows)} rows")


def required_files(directory):
    return [directory / "hnsw_build.csv", directory / "retrieval_ch2.csv"] + [
        directory / f"vamana_lvc{suffix}_compression.csv"
        for _, suffix in CHAINS
    ]


def summarize_dataset(directory):
    dataset = directory.name
    hnsw_rows = read_csv(directory / "hnsw_build.csv")
    if len(hnsw_rows) != 12:
        raise ValueError(f"{dataset}: expected 12 HNSW codec/chain rows, got {len(hnsw_rows)}")
    expected = {(codec, chain) for codec in CODECS for chain, _ in CHAINS}
    found = {(row["Algorithm"].lower(), int(row["ChainMaxLength"])) for row in hnsw_rows}
    if found != expected:
        raise ValueError(f"{dataset}: HNSW codec/chain configurations are incomplete")

    compression = []
    for row in hnsw_rows:
        compression.append(dict(dataset=dataset, graph="HNSW", codec=row["Algorithm"].lower(),
            chain_max=row["ChainMaxLength"], target_roots=row["InitialRoots"],
            roots=row["Roots"], max_depth=row["MaxDepth"],
            average_depth=row["AverageDepth"], average_nonroot_depth=row["AverageNonrootDepth"],
            mean_reference_l2=row["MeanReferenceL2"],
            p50_reference_l2=row["P50ReferenceL2"], p95_reference_l2=row["P95ReferenceL2"],
            payload_bytes=row["PayloadBytes"],
            patch_bytes=row["F32PatchBytes"], raw_f32_bytes=row["RawF32Bytes"],
            graph_bytes=row["GraphMetadataBytes"], metadata_bytes=0,
            state_cache_bytes=row["StateCacheBytes"],
            raw_f32_index_bytes=row["RawF32IndexBytes"],
            compressed_index_bytes=row["CompressedIndexBytes"],
            payload_ratio_f32=row["PayloadRatioF32"],
            full_index_ratio_f32=row["FullIndexRatioF32"], record_file_materialized=1))

    for chain, suffix in CHAINS:
        rows = read_csv(directory / f"vamana_lvc{suffix}_compression.csv")
        if len(rows) != 3 or {row["codec"].lower() for row in rows} != set(CODECS):
            raise ValueError(f"{dataset}: incomplete Vamana compression at chain={chain}")
        for row in rows:
            compression.append(dict(dataset=dataset, graph="Vamana", codec=row["codec"].lower(),
                chain_max=chain, target_roots=row["roots"], roots=row["roots"],
                max_depth=row["max_depth"], average_depth=row["average_depth"],
                average_nonroot_depth=row["average_nonroot_depth"],
                mean_reference_l2=row["mean_reference_l2"],
                p50_reference_l2=row["p50_reference_l2"], p95_reference_l2=row["p95_reference_l2"],
                payload_bytes=row["payload_bytes"], patch_bytes=row["patch_bytes"],
                raw_f32_bytes=row["raw_f32_bytes"], graph_bytes=row["graph_bytes"],
                metadata_bytes=row["metadata_bytes"], state_cache_bytes=0,
                raw_f32_index_bytes=row["raw_f32_index_bytes"],
                compressed_index_bytes=row["compressed_index_bytes"],
                payload_ratio_f32=row["payload_ratio_f32"],
                full_index_ratio_f32=row["full_index_ratio_f32"],
                record_file_materialized=row["record_file_materialized"]))

    retrieval = read_csv(directory / "retrieval_ch2.csv")
    expected_curves = {(graph, codec, tls) for graph in ("HNSW", "Vamana")
                       for codec in CODECS for tls in ("0", "1")}
    found_curves = {(row["graph"], row["codec"], row["tls"]) for row in retrieval
                    if int(row["k"]) == 1}
    if not expected_curves <= found_curves:
        raise ValueError(f"{dataset}: incomplete K=1 retrieval curves")
    if any(row["dataset"] != dataset for row in retrieval):
        raise ValueError(f"{dataset}: retrieval CSV contains another dataset")
    return compression, retrieval


def align_latency(retrieval, datasets):
    aligned = []
    for dataset in datasets:
        for codec in CODECS:
            for tls in ("0", "1"):
                pair = {graph: [row for row in retrieval if row["dataset"] == dataset
                                and row["graph"] == graph and row["codec"] == codec
                                and int(row["k"]) == 1 and row["tls"] == tls]
                        for graph in ("HNSW", "Vamana")}
                targets = sorted({round(float(row["recall"]), 6)
                                  for rows in pair.values() for row in rows})
                for target in targets:
                    eligible = {graph: [row for row in rows
                                        if float(row["recall"]) + 1e-9 >= target]
                                for graph, rows in pair.items()}
                    if not all(eligible.values()):
                        continue
                    best = {graph: min(rows, key=lambda row: float(row["mean_us"]))
                            for graph, rows in eligible.items()}
                    hnsw, vamana = best["HNSW"], best["Vamana"]
                    aligned.append(dict(dataset=dataset, codec=codec, k=1, tls=tls,
                        target_recall=target, hnsw_recall=hnsw["recall"],
                        hnsw_ef=hnsw["search_width"], hnsw_mean_us=hnsw["mean_us"],
                        vamana_recall=vamana["recall"], vamana_L=vamana["search_width"],
                        vamana_mean_us=vamana["mean_us"],
                        vamana_over_hnsw_latency=float(vamana["mean_us"]) / float(hnsw["mean_us"])))
    return aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/data/cross_graph_transfer",
                        help="Root containing the per-dataset experiment directories")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Publish summary CSVs here as well as under --root (default: %(default)s)")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"result directory does not exist: {root}")

    compression, retrieval, completed = [], [], []
    directories = sorted((p for p in root.iterdir() if p.is_dir()),
                         key=lambda p: (DATASET_ORDER.index(p.name) if p.name in DATASET_ORDER
                                        else len(DATASET_ORDER), p.name))
    for directory in directories:
        missing = [path.name for path in required_files(directory) if not path.is_file()]
        if missing:
            print(f"SKIP {directory.name}: missing {', '.join(missing)}")
            continue
        try:
            dataset_compression, dataset_retrieval = summarize_dataset(directory)
        except (KeyError, ValueError) as error:
            print(f"SKIP {directory.name}: {error}")
            continue
        compression.extend(dataset_compression)
        retrieval.extend(dataset_retrieval)
        completed.append(directory.name)
        print(f"INCLUDE {directory.name}: {len(dataset_compression)} compression rows, "
              f"{len(dataset_retrieval)} retrieval rows")
    if not completed:
        parser.error("no complete cross-graph datasets found")

    alpha_rows = [row for row in retrieval
                  if row["tls"] == "1" and abs(float(row["alpha"]) - 0.2) < 1e-6]
    aligned = align_latency(retrieval, completed)
    # Keep the historical build/data outputs and publish the same data to the
    # versioned results directory used by plotting and paper analysis.
    for destination in dict.fromkeys((root, args.output_dir.resolve())):
        destination.mkdir(parents=True, exist_ok=True)
        write_csv(destination / "compression_summary.csv", compression)
        write_csv(destination / "retrieval_summary.csv", retrieval)
        write_csv(destination / "retrieval_alpha_0p2.csv", alpha_rows, list(retrieval[0]))
        write_csv(destination / "recall_aligned_latency.csv", aligned)
    print("Complete datasets:", ", ".join(completed))


if __name__ == "__main__":
    main()
