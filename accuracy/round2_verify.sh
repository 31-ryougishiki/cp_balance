#!/usr/bin/env bash
# Round-2 verification driver: static gates -> C acceptance -> B equivalence -> optional A/B.
#   bash accuracy/round2_verify.sh [--steps 0,1,2,3] [--dry-run] [...]
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/round2_verify.py" "$@"
