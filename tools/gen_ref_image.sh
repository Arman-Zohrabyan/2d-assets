#!/usr/bin/env bash
# One PNG via OpenAI image generation through the Codex CLI, with attached reference images.
# Same call as the codex-image MCP server (reference_images), usable from scripts in parallel.
# Usage: tools/gen_ref_image.sh OUT.png SIZE PROMPT_FILE [REF.png ...]
set -uo pipefail
out="$1"; size="$2"; prompt_file="$3"; shift 3
args=()
for r in "$@"; do args+=("--image=$r"); done
dir="$(mktemp -d "${TMPDIR:-/tmp}/codex-image-XXXXXX")"
log="$dir.log"
trap 'rm -rf "$dir"' EXIT
refs=""
[ $# -gt 0 ] && refs="$# reference image(s) are attached (image 1 to image $#, in order). Pass them to image generation as visual references."
instr="Use your built-in image generation (\$imagegen) to create exactly one image.
Size: $size.
Use a transparent background.
$refs
Image description: $(cat "$prompt_file")
Save it as image.png in the current directory. Do not create or change any other files."
codex exec --skip-git-repo-check -C "$dir" -s workspace-write -m "${CODEX_IMAGE_MODEL:-gpt-6.1-sol}" \
  -c 'model_reasoning_effort="low"' "${args[@]}" -- "$instr" </dev/null >"$log" 2>&1
png="$(find "$dir" -maxdepth 1 -iname '*.png' | head -1)"
if [ -z "$png" ]; then
  reason="$(grep -o 'safety_violations=\[[a-z_, ]*\]\|moderation_blocked' "$log" | head -1)"
  echo "FAILED $out ${reason:-see $log}"; exit 1
fi
mkdir -p "$(dirname "$out")"; cp "$png" "$out"; echo "Saved $out"
