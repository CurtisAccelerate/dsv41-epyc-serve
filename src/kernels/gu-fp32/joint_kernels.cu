// One-user cubin. The 4-parameter symbols are the SPEC=6 fold bodies.
// SPEC is a constexpr inside those templates, not a fifth template argument
// and not a call into another kernel.
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"

static_assert(sizeof(MoeCoopParams) == 344, "MoeCoopParams ABI");

template __global__ void exl3_moe_coop_ns::exl3_moe_coop_a_kernel<2, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_a_kernel<3, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_b_kernel<2, 2, true, false>(const MoeCoopParams);
template __global__ void exl3_moe_coop_ns::exl3_moe_coop_b_kernel<3, 2, true, false>(const MoeCoopParams);
