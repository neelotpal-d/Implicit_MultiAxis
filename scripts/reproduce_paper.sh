#!/usr/bin/env bash
# Reproduce the two paper figures end-to-end.
#
#   ./scripts/reproduce_paper.sh             — run both
#   ./scripts/reproduce_paper.sh fertility   — just the support-free figure
#   ./scripts/reproduce_paper.sh tbracket    — just the toolpath-alignment figure
#
# Requires pixi. Uses the seeds in examples/configs/*.json. Outputs land
# in examples/outputs/<dir>/{checkpoints,losses}/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

target="${1:-all}"

run_fertility() {
    echo "=== Fertility (support-free) ==="
    pixi run support-free
    echo "=== Rendering fertility figure ==="
    pixi run python examples/display_latest_checkpoint.py examples/configs/support_free_config.json
}

run_tbracket() {
    echo "=== T-bracket (toolpath alignment) ==="
    pixi run toolpath-alignment
    echo "=== Rendering T-bracket figure ==="
    pixi run python examples/display_latest_checkpoint.py examples/configs/toolpath_alignment_config.json
}

case "$target" in
    fertility) run_fertility ;;
    tbracket)  run_tbracket ;;
    all)       run_fertility; run_tbracket ;;
    *)         echo "usage: $0 [fertility|tbracket|all]" >&2; exit 2 ;;
esac
