#!/bin/sh
set -eu
cd "$(dirname "$0")"
inc=${EXL3_INCLUDE:-}
header="$inc/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"
if [ -z "$inc" ] || [ ! -f "$header" ]; then
  echo "Set EXL3_INCLUDE to an ExLlamaV3 tree that contains exllamav3_ext/quant/exl3_moe_coop_kernel.cuh" >&2
  echo "That header is not in this image. These launchers do not compile without it." >&2
  exit 1
fi
nvcc=${NVCC:-nvcc}
arch=${CUDA_ARCH:-sm_80}
"$nvcc" -shared -Xcompiler -fPIC -arch="$arch" -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=177,20012 -DKBITS=3 -I"$inc" launch.cu -o libgu_rows.so
"$nvcc" -shared -Xcompiler -fPIC -arch="$arch" -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=177,20012 -DKBITS=3 -I"$inc" joint_launch.cu -o libgu_joint.so
"$nvcc" -shared -Xcompiler -fPIC -arch="$arch" -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=177,20012 joint_kernels.cu -o libgu_symbols.so
echo "built libgu_rows.so libgu_joint.so libgu_symbols.so"
