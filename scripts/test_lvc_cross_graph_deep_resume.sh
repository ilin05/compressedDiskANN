#!/usr/bin/env bash
set -euo pipefail

# DEEP only. Reuse graphs, PQ, normalized inputs and complete experiment rows.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -u "$SCRIPT_DIR/resume_lvc_cross_graph_deep.py" "$@"
