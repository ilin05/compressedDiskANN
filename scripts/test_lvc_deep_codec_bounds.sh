#!/usr/bin/env bash
set -euo pipefail

# Standalone regression: normalized F32 with tiny nonzero coordinates, all chains.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${OUT:-$DISKANN_DIR/build/data/lvc_deep_codec_bounds}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
flags=(-std=c++17 -O1 -g -fopenmp -fno-omit-frame-pointer)
if [[ "${SANITIZE:-1}" == 1 ]]; then
  flags+=(-fsanitize=address -fno-pie -no-pie)
fi
# LeakSanitizer is unavailable in some restricted/ptraced execution environments.
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:handle_segv=0}"
for app in convert_lvc_vamana verify_lvc_vamana; do
  "${CXX:-g++}" "${flags[@]}" -I "$DISKANN_DIR/include" \
    "$DISKANN_DIR/apps/$app.cpp" \
    "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
    -o "$OUT/$app"
done
PYTHONDONTWRITEBYTECODE=1 python3 - "$SCRIPT_DIR" "$OUT" <<'PY'
import math, pathlib, random, struct, subprocess, sys
sys.path.insert(0, sys.argv[1])
from resume_lvc_cross_graph_deep import compression_check, read_csv, write_csv
out = pathlib.Path(sys.argv[2]); n, dim = 512, 96
rng = random.Random(20261027)
with (out / 'train.fbin').open('wb') as f:
    f.write(struct.pack('<II', n, dim))
    for i in range(n):
        vector = [rng.gauss(0, 1) for _ in range(dim)]
        vector[0] = 0 if i < 5 else 1e-8
        vector[1] = -1e-10
        vector[2] = 0
        norm = math.sqrt(sum(x*x for x in vector))
        f.write(struct.pack('<%df' % dim, *(x/norm for x in vector)))
with (out / 'graph').open('wb') as f:
    f.write(struct.pack('<QIIQ', 24 + n * 68, 16, 0, 0))
    for i in range(n):
        f.write(struct.pack('<I16I', 16, *((i+j+1) % n for j in range(16))))
rows = []
for chain in (2, 4, 8, -1):
    for threads in (1, 4):
        prefix = out / f'ch{chain}_t{threads}'
        command = [out / 'convert_lvc_vamana', out / 'graph', out / 'train.fbin', prefix,
                   '--chain_max', chain, '--threads', threads]
        if chain != 2:
            command.append('--compression_only')
        log = out / f'ch{chain}_t{threads}.log'
        with log.open('w') as f:
            result = subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT, timeout=60)
        if result.returncode:
            raise SystemExit(f'FAIL chain={chain} threads={threads}: see {log}')
        compression_check(prefix, n, dim, chain)
    one, four = [read_csv(out / f'ch{chain}_t{t}_compression.csv') for t in (1, 4)]
    if one != four or (out / f'ch{chain}_t1_forest.csv').read_bytes() != (out / f'ch{chain}_t4_forest.csv').read_bytes():
        raise SystemExit(f'FAIL chain={chain}: threads changed forest/bytes/statistics')
    rows.extend(dict(chain_max=chain, **r) for r in four)
    print(f'PASS normalized F32 decimal bounds: chain={chain}, threads=1/4', flush=True)
for codec in ('raw', 'dexor', 'gorilla', 'elf'):
    subprocess.run([str(out / 'verify_lvc_vamana'), str(out / 'train.fbin'),
                    str(out / f'ch2_t4_{codec}.data')], check=True, timeout=60)
write_csv(out / 'summary.csv', rows)
PY
