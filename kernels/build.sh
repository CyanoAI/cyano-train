#!/bin/bash
# Build the HIP kernels for the target DCU architecture.
# Override HIPCC and ARCH for your environment.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HIPCC="${HIPCC:-/opt/dtk-26.04/bin/hipcc}"
ARCH="${ARCH:-gfx906}"

mkdir -p "$HERE/bin" "$HERE/logs"
for name in rmsnorm_residual rope_qk_layout swiglu; do
  echo "building $name for $ARCH"
  "$HIPCC" --offload-arch="$ARCH" -shared -fPIC \
    "$HERE/src/$name.hip" -o "$HERE/bin/$name.so" \
    > "$HERE/logs/compile_$name.log" 2>&1
done
echo "kernels written to $HERE/bin"
