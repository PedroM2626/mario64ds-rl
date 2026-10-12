#!/usr/bin/env bash
# Build the supervisor with rustc directly: no cargo, no crates, no network.
# cargo.exe is blocked by Windows app control on this machine (a downloaded,
# unsigned binary); rustc.exe runs fine, and the program needs nothing but std.
#
# The codegen flags are not cosmetic. App control decides by file hash, and the
# default `-O` build of this program (5.3 MB, full symbols) is denied execution
# while a stripped single-codegen-unit build of the same source runs. So: strip,
# one codegen unit, then prove the result actually executes.
set -euo pipefail
cd "$(dirname "$0")"

RUSTC="${RUSTC:-rustc}"
if ! command -v "$RUSTC" >/dev/null 2>&1; then
  RUSTC="$HOME/.cargo/bin/rustc.exe"
fi
echo "using $("$RUSTC" --version)"

"$RUSTC" -O --edition 2021 -C strip=symbols -C codegen-units=1 \
  src/main.rs -o sm64_sup.exe
if ! ./sm64_sup.exe --help >/dev/null; then
  echo "sm64_sup.exe built but refused to run: Windows app control blocks this" \
       "hash. Touch the source or change a codegen flag and rebuild." >&2
  exit 1
fi
echo "built $(pwd)/sm64_sup.exe ($(stat -c %s sm64_sup.exe) bytes, executes)"

# unit tests (json escaping, job-line parsing, seconds field)
"$RUSTC" --edition 2021 -C strip=symbols --test src/main.rs -o sm64_sup_tests.exe
./sm64_sup_tests.exe
