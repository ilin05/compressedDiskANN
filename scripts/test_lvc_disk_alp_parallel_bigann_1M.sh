#!/usr/bin/env bash
# Verify identical ALP records and deterministic beamwidth-one retrieval.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
index=${INDEX_PREFIX:-"$data/disk_index_bigann_1M_R32_L50_A1.2"}
query=${QUERY_FILE:-"$data/bigann_query.fbin"}
gt=${GT_FILE:-null}
nodes=${CACHE_NODES:-1024}
threads=${ALP_ENCODE_THREADS:-4}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_alp_parallel_results/bigann_1M_$(date -u +%Y%m%dT%H%M%SZ)"}
[[ "$threads" =~ ^[0-9]+$ && "$threads" -gt 1 ]] || { echo "ALP_ENCODE_THREADS must exceed 1" >&2; exit 1; }
mkdir -p "$out"
if [[ -e "$out/alp_t1.log" || -e "$out/alp_t${threads}.log" ]]; then
    echo "Existing results in $out; choose another OUTPUT_DIR." >&2
    exit 1
fi
echo "Results: $out"
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for n in 1 "$threads"; do
    prefix="$out/alp_t$n"
    "$build/apps/search_disk_index" --data_type float --dist_fn l2 --index_path_prefix "$index" \
        --query_file "$query" --gt_file "$gt" --cache_codec alp --verify_cache \
        --num_nodes_to_cache "$nodes" --alp_encode_threads "$n" \
        -K 1 -L 10 -W 1 -T 1 -R 1 --result_path "$prefix" > "$prefix.log" 2>&1
    cp "${index}_K1_T1_C${nodes}_search_result.csv" "$prefix.csv"
    grep -E 'ALP_CACHE_TIMING|ALP_CACHE_VERIFY|CACHE_BUDGET|LVC_LOAD_TIMING' "$prefix.log"
done
python3 - "$out" "$threads" "$nodes" <<'PY'
import pathlib,re,struct,sys
out=pathlib.Path(sys.argv[1]); threads=sys.argv[2]; nodes=int(sys.argv[3])
logs=[(out/('alp_t'+n+'.log')).read_text() for n in ('1',threads)]
def one(pattern,text):
    m=re.search(pattern,text)
    if not m: raise SystemExit('Missing expected log: '+pattern)
    return m.groups()
cache=[one(r'Cache node list: count=(\d+) hash=(\d+)',s) for s in logs]
assert cache[0]==cache[1] and int(cache[0][0])==nodes, 'cache nodes differ'
verify=[one(r'ALP_CACHE_VERIFY vectors=(\d+) coordinates=(\d+) exact_coordinates=(\d+) max_abs_error=([\deE.+-]+) encoded_hash=(\d+)',s) for s in logs]
assert verify[0]==verify[1] and int(verify[0][0])==nodes, 'ALP verification differs'
budget=[one(r'CACHE_BUDGET .*?vector_payload_bytes=(\d+)',s) for s in logs]
assert budget[0]==budget[1], 'compressed payload sizes differ'
for suffix in ('idx_uint32.bin','dists_float.bin'):
    a=(out/('alp_t1_10_'+suffix)).read_bytes(); b=(out/('alp_t'+threads+'_10_'+suffix)).read_bytes()
    if a!=b:
        pos=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        item=(pos-8)//4
        fmt='<I' if suffix.startswith('idx') else '<f'
        if pos>=8 and 8+4*(item+1)<=min(len(a),len(b)):
            raise SystemExit('FAIL %s query=%d serial=%s parallel=%s' % (suffix,item,struct.unpack_from(fmt,a,8+4*item)[0],struct.unpack_from(fmt,b,8+4*item)[0]))
        raise SystemExit('FAIL %s byte=%d' % (suffix,pos))
    print('PASS serial/parallel exact',suffix)
print('PASS cache hash=%s vectors=%s coordinates=%s bitwise_coordinates=%s max_abs_error=%s encoded_hash=%s payload_bytes=%s' %
      (cache[0][1],*verify[0],budget[0][0]))
print('Results:',out)
PY
