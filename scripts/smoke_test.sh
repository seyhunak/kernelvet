#!/usr/bin/env bash
# Smoke-test an installed `kernelvet` on a machine WITHOUT an AMD GPU (e.g. CI).
# It checks that the CLI works end to end and that nothing claims GPU
# verification that did not happen. Usage: scripts/smoke_test.sh [path/to/kernelvet]
set -euo pipefail

KERNELVET="${1:-kernelvet}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export KERNELVET_CONFIG_DIR="$WORK/cfg" KERNELVET_DATA_DIR="$WORK/data"

tree_digest() {  # content hash of every file under $1
  python3 - "$1" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]); h = hashlib.sha256()
for f in sorted(p for p in root.rglob("*") if p.is_file()):
    h.update(f.relative_to(root).as_posix().encode()); h.update(f.read_bytes())
print(h.hexdigest())
PY
}

expect_exit() {  # expect_exit "<allowed codes>" cmd...
  local allowed="$1"; shift
  set +e; "$@"; local rc=$?; set -e
  for c in $allowed; do [[ $rc == "$c" ]] && return 0; done
  echo "FAIL: '$*' exited $rc, expected one of: $allowed" >&2; exit 1
}

echo "== version / help"
"$KERNELVET" --version
"$KERNELVET" --help > /dev/null

echo "== diagnose (JSON on stdout must be clean)"
expect_exit "1 3" "$KERNELVET" diagnose --format json --no-history > "$WORK/diag.json" 2> "$WORK/diag.err"
python3 - "$WORK/diag.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["kind"] == "diagnosis", r["kind"]
assert r["status"] != "HEALTHY", "a GPU-less host must not be HEALTHY"
bad = [c["id"] for c in r["checks"] if c["evidence"] == "executed" and c["status"] == "PASS"]
assert not bad, f"executed PASS on a CPU-only host: {bad}"
print("diagnosis:", r["status"], "-", len(r["checks"]), "checks, none executed")
PY

echo "== other read-only commands"
expect_exit "0 1 3" "$KERNELVET" doctor > /dev/null
expect_exit "0 1 3" "$KERNELVET" compatibility --format json > "$WORK/compat.json"
expect_exit "1" "$KERNELVET" gpu list --json > "$WORK/gpus.json"  # 1 = no AMD GPU
"$KERNELVET" system info --json > /dev/null
"$KERNELVET" config show > /dev/null
for k in diagnosis compatibility migration benchmark; do "$KERNELVET" schema "$k" > /dev/null; done

echo "== migration on a copy of the fixture project"
cp -R "$HERE/tests/fixtures/cuda/vector_project" "$WORK/cuda"
before="$(tree_digest "$WORK/cuda")"
expect_exit "0 3" "$KERNELVET" migrate analyze "$WORK/cuda" --format json > "$WORK/analysis.json"
expect_exit "0 3" "$KERNELVET" migrate plan "$WORK/cuda" > /dev/null
if command -v hipify-perl > /dev/null || command -v hipify-clang > /dev/null; then
  echo "HIPIFY present; skipping the 'unavailable' check"
else
  expect_exit "4" "$KERNELVET" migrate convert "$WORK/cuda" -o "$WORK/hip"
fi
after="$(tree_digest "$WORK/cuda")"
[[ "$before" == "$after" ]] || { echo "FAIL: the CUDA source tree was modified" >&2; exit 1; }

echo "smoke test passed"
