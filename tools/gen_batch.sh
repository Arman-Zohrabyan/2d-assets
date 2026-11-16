#!/usr/bin/env bash
# Run many generations in parallel. Each line of JOBS: OUT.png|SIZE|PROMPT_FILE|REF1|REF2...
# Usage: tools/gen_batch.sh JOBS [PARALLEL=4]   (lines whose OUT already exists are skipped)
set -uo pipefail
jobs="$1"; par="${2:-4}"
here="$(cd "$(dirname "$0")" && pwd)"
grep -v '^\s*#' "$jobs" | grep -v '^\s*$' | while IFS= read -r line; do
  out="${line%%|*}"
  [ -f "$out" ] && continue
  printf '%s\0' "$line"
done | xargs -0 -P "$par" -I{} bash -c 'IFS="|" read -r -a f <<< "$1"; "$2/gen_ref_image.sh" "${f[@]}"' _ {} "$here"
