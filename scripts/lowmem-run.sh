#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Keep local builds predictable on small production hosts. Every setting can
# be overridden by the caller without changing the normal build path.
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=${NODE_OLD_SPACE_MB:-1792}}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export npm_config_jobs="${npm_config_jobs:-1}"
export MAKEFLAGS="${MAKEFLAGS:--j1}"

if [[ -w /proc/self/oom_score_adj ]]; then
    echo "${BUILD_OOM_SCORE_ADJ:-750}" > /proc/self/oom_score_adj 2>/dev/null || true
fi

if [[ -n "${BUILD_VMEM_MB:-}" ]]; then
    ulimit -Sv "$((BUILD_VMEM_MB * 1024))"
fi

cmd=("$@")
if [[ ${#cmd[@]} -eq 0 ]]; then
    cmd=(npm run build)
fi

runner=()
if command -v ionice >/dev/null 2>&1; then
    runner+=(ionice -c3)
fi
if command -v nice >/dev/null 2>&1; then
    runner+=(nice -n "${BUILD_NICE:-19}")
fi
if [[ -n "${BUILD_CPUSET:-}" ]] && command -v taskset >/dev/null 2>&1; then
    runner+=(taskset -c "$BUILD_CPUSET")
fi

exec "${runner[@]}" "${cmd[@]}"
