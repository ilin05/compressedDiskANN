#!/usr/bin/env bash
# Build the SAME benchmark against test_bigann native graph/distance code.
# Usage: RAW_REPO=/path/to/test_bigann_checkout bash scripts/build_dynamic_vamana_raw_baseline.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RAW_REPO:?Set RAW_REPO to a test_bigann checkout}"
RAW_BUILD_DIR="${RAW_BUILD_DIR:-$RAW_REPO/build}"
DRIVER_BUILD_DIR="${DRIVER_BUILD_DIR:-$RAW_BUILD_DIR/dynamic_vamana_driver}"
if [[ "$(git -C "$RAW_REPO" branch --show-current)" != test_bigann ]]; then
  echo 'RAW_REPO must be on test_bigann' >&2; exit 1
fi
mkdir -p "$DRIVER_BUILD_DIR"
# The only source change in the Raw checkout is a friend declaration, allowing
# static import/statistics without changing data members or graph/distance code.
# Rebuild the library and driver with the SAME header definition (ODR-safe).
python3 - "$RAW_REPO/include/index.h" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
if 'friend class DynamicVamanaExperiment;' not in s:
    if '  private:' not in s: raise RuntimeError('Cannot locate Index private section')
    p.write_text(s.replace('  private:', '  private:\n    friend class DynamicVamanaExperiment;', 1))
    print('Added import-adapter friend declaration:', p)
PY
cmake -G "Unix Makefiles" -S "$RAW_REPO" -B "$RAW_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$RAW_BUILD_DIR" --target build_memory_index -j "${BUILD_JOBS:-8}"
# Reuse the native tool's complete compile/link flags (including static MKL,
# libaio, Boost and tcmalloc). A plain -ldiskann is insufficient for static builds.
python3 - "$RAW_REPO" "$RAW_BUILD_DIR" "$DRIVER_BUILD_DIR" "$SCRIPT_DIR/../apps/test_dynamic_lvc_vamana.cpp" <<'PYCODE'
from pathlib import Path
import shlex, subprocess, sys
repo, build, driver, source = map(lambda x: Path(x).resolve(), sys.argv[1:])
flags_file = build/'apps/CMakeFiles/build_memory_index.dir/flags.make'
settings = {}
for line in flags_file.read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        settings[key] = shlex.split(value)
link = shlex.split((build/'apps/CMakeFiles/build_memory_index.dir/link.txt').read_text())
compiler = link[0]
obj = driver/'dynamic_driver.o'
flags = sum((settings.get(k, []) for k in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS')), [])
command = [compiler, '-I'+str(repo/'include'), *flags,
           '-c', str(source), '-o', str(obj)]
subprocess.run(command, check=True)
indices = [i for i, arg in enumerate(link) if arg.endswith('build_memory_index.cpp.o')]
if len(indices) != 1: raise RuntimeError('Cannot identify native tool object in link.txt')
link[indices[0]] = str(obj)
link[link.index('-o')+1] = str(driver/'test_dynamic_lvc_vamana_raw')
subprocess.run(link, cwd=build/'apps', check=True)
PYCODE
printf 'RAW_BINARY=%s\n' "$DRIVER_BUILD_DIR/test_dynamic_lvc_vamana_raw"
