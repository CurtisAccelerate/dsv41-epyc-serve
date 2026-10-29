#!/usr/bin/env bash
# Rebuild into an isolated output directory; never overwrite the pinned serving libraries.
set -euo pipefail
destination=${1:-/tmp/dsv41-gates-rebuilt}
mkdir -p "$destination/include/exllamav3_ext/quant"
cp -a /opt/dsv41-build/exllamav3/exllamav3/exllamav3_ext/. "$destination/include/exllamav3_ext/"
cp /opt/dsv41/source/kernels/exl3/*.cuh "$destination/include/exllamav3_ext/quant/"
torch_include=$(python3 -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "include")')
for bits in 2 3; do
  if [ "$bits" = 3 ]; then library=libgu_rows.so; else library=libgu_k2.so; fi
  nvcc -shared -Xcompiler -fPIC -arch=sm_80 -std=c++17 -O3 --use_fast_math \
    --expt-relaxed-constexpr -diag-suppress=177,20012 -DKBITS="$bits" \
    -I"$destination/include" -I"$torch_include" -I"$torch_include/torch/csrc/api/include" \
    /opt/dsv41/source/kernels/gu-fp32/launch.cu \
    -o "$destination/$library"
done
sha256sum "$destination"/*.so
echo 'Rebuilt outputs require operator correctness/performance qualification before replacing serving pins.'
