// Host launcher for a coop cubin. Same MoeCoopParams fill as the fold launcher.
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include "exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"

static_assert(sizeof(MoeCoopParams) == 344, "MoeCoopParams ABI");

struct GuLaunch {
    void* x;
    void* sel;
    void* rw;
    void* g_trellis;
    void* g_suh;
    void* g_svh;
    void* u_trellis;
    void* u_suh;
    void* u_svh;
    void* d_trellis;
    void* d_suh;
    void* d_svh;
    void* had_g;
    void* had_u;
    void* gu_g;
    void* gu_u;
    void* act_out;
    void* d_out;
    void* ctr;
    void* out;
    int64_t x_stride;
    int64_t out_stride;
    int64_t bsz;
    int64_t topk;
    int64_t H;
    int64_t Hi;
    int64_t I;
    int64_t Ho;
    int64_t H_out;
    int64_t min_expert;
    int64_t max_expert;
    int64_t act;
    int64_t gated;
    int64_t gu_f32;
    int64_t slots_max;
    int64_t rows_max;
    int64_t n_local;
    double act_limit;
};

static const char* SYM_A2 = "_ZN16exl3_moe_coop_ns22exl3_moe_coop_a_kernelILi2ELi2ELb1ELb0EEEv13MoeCoopParams";
static const char* SYM_A3 = "_ZN16exl3_moe_coop_ns22exl3_moe_coop_a_kernelILi3ELi2ELb1ELb0EEEv13MoeCoopParams";
static const char* SYM_B2 = "_ZN16exl3_moe_coop_ns22exl3_moe_coop_b_kernelILi2ELi2ELb1ELb0EEEv13MoeCoopParams";
static const char* SYM_B3 = "_ZN16exl3_moe_coop_ns22exl3_moe_coop_b_kernelILi3ELi2ELb1ELb0EEEv13MoeCoopParams";
static const char* SYM_ROT = "_ZN16exl3_moe_coop_ns24exl3_moe_coop_rot_kernelE13MoeCoopParams";

struct Loaded {
    CUmodule mod = nullptr;
    CUfunction rot = nullptr;
    CUfunction a = nullptr;
    CUfunction b = nullptr;
    int smem_a = 0;
    int smem_b = 0;
};

static std::unordered_map<std::string, Loaded> g_loaded;

static int fail(CUresult st, const char* what) {
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(st, &name);
    cuGetErrorString(st, &desc);
    std::fprintf(stderr, "joint_launch %s: %s %s\n", what, name ? name : "?", desc ? desc : "");
    return 1;
}

static int bind_context(CUstream stream) {
    CUcontext ctx = nullptr;
    CUresult st = cuStreamGetCtx(stream, &ctx);
    if (st != CUDA_SUCCESS || ctx == nullptr) {
        int ordinal = 0;
        cudaGetDevice(&ordinal);
        CUdevice dev;
        st = cuDeviceGet(&dev, ordinal);
        if (st != CUDA_SUCCESS) return fail(st, "cuDeviceGet");
        st = cuDevicePrimaryCtxRetain(&ctx, dev);
        if (st != CUDA_SUCCESS) return fail(st, "primary ctx");
    }
    st = cuCtxSetCurrent(ctx);
    if (st != CUDA_SUCCESS) return fail(st, "cuCtxSetCurrent");
    return 0;
}

static int load_module(const char* path, int bits, CUstream stream, Loaded** out) {
    std::string key = std::string(path) + ":" + std::to_string(bits);
    auto found = g_loaded.find(key);
    if (found != g_loaded.end()) {
        *out = &found->second;
        return 0;
    }
    if (bind_context(stream)) return 1;
    Loaded item;
    CUresult st = cuModuleLoad(&item.mod, path);
    if (st != CUDA_SUCCESS) return fail(st, "cuModuleLoad");
    st = cuModuleGetFunction(&item.rot, item.mod, SYM_ROT);
    if (st != CUDA_SUCCESS) return fail(st, SYM_ROT);
    const char* sym_a = bits == 2 ? SYM_A2 : SYM_A3;
    const char* sym_b = bits == 2 ? SYM_B2 : SYM_B3;
    st = cuModuleGetFunction(&item.a, item.mod, sym_a);
    if (st != CUDA_SUCCESS) return fail(st, sym_a);
    st = cuModuleGetFunction(&item.b, item.mod, sym_b);
    if (st != CUDA_SUCCESS) return fail(st, sym_b);
    item.smem_a = bits == 2 ? exl3_moe_coop_ns::smem_a_bytes<2>(5120) : exl3_moe_coop_ns::smem_a_bytes<3>(5120);
    item.smem_b = bits == 2 ? exl3_moe_coop_ns::smem_b_bytes<2>() : exl3_moe_coop_ns::smem_b_bytes<3>();
    auto placed = g_loaded.emplace(key, item);
    *out = &placed.first->second;
    std::fprintf(stderr, "joint_launch loaded bits=%d smem_a=%d smem_b=%d %s\n", bits, item.smem_a, item.smem_b, path);
    return 0;
}

extern "C" int joint_params_bytes() {
    return (int)sizeof(MoeCoopParams);
}

extern "C" int joint_cubin_launch(const char* cubin, int bits, const GuLaunch* in, void* stream_ptr) {
    if (bits != 2 && bits != 3) {
        std::fprintf(stderr, "joint_launch bits %d\n", bits);
        return 1;
    }
    CUresult init = cuInit(0);
    if (init != CUDA_SUCCESS) return fail(init, "cuInit");
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    Loaded* mod = nullptr;
    if (load_module(cubin, bits, stream, &mod)) return 1;

    MoeCoopParams p = {};
    p.x = reinterpret_cast<const half*>(in->x);
    p.x_stride = (int)in->x_stride;
    p.sel = reinterpret_cast<const int64_t*>(in->sel);
    p.rw = reinterpret_cast<const half*>(in->rw);
    p.bsz = (int)in->bsz;
    p.topk = (int)in->topk;
    p.H = (int)in->H;
    p.Hi = (int)in->Hi;
    p.I = (int)in->I;
    p.Ho = (int)in->Ho;
    p.H_out = (int)in->H_out;
    p.min_expert = (int)in->min_expert;
    p.max_expert = (int)in->max_expert;
    p.g_trellis = reinterpret_cast<const int64_t*>(in->g_trellis);
    p.g_suh = reinterpret_cast<const int64_t*>(in->g_suh);
    p.g_svh = reinterpret_cast<const int64_t*>(in->g_svh);
    p.u_trellis = reinterpret_cast<const int64_t*>(in->u_trellis);
    p.u_suh = reinterpret_cast<const int64_t*>(in->u_suh);
    p.u_svh = reinterpret_cast<const int64_t*>(in->u_svh);
    p.d_trellis = reinterpret_cast<const int64_t*>(in->d_trellis);
    p.d_suh = reinterpret_cast<const int64_t*>(in->d_suh);
    p.d_svh = reinterpret_cast<const int64_t*>(in->d_svh);
    p.g_bias = nullptr;
    p.u_bias = nullptr;
    p.d_bias = nullptr;
    p.act = (int)in->act;
    p.act_limit = (float)in->act_limit;
    p.gated = in->gated != 0;
    p.had_g = reinterpret_cast<half*>(in->had_g);
    p.had_u = reinterpret_cast<half*>(in->had_u);
    p.a_global = p.bsz > 1;
    p.gu_g = in->gu_g;
    p.gu_u = in->gu_u;
    p.gu_f32 = in->gu_f32 != 0;
    p.act_out = reinterpret_cast<half*>(in->act_out);
    p.d_out = reinterpret_cast<float*>(in->d_out);
    p.slots_max = (int)in->slots_max;
    p.rows_max = (int)in->rows_max;
    p.n_local = (int)in->n_local;
    p.ctr_a_len = p.slots_max * (p.I / 128);
    p.ctr_b_len = p.rows_max * (p.Ho / 128);
    p.ctr_a = reinterpret_cast<int*>(in->ctr);
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.ksplit_a = 1;
    p.ksplit_b = 1;
    p.dbg = 0;
    p.sh_gate_n = 0;
    p.sh_out = nullptr;
    p.sh_gate_w = nullptr;
    p.out = reinterpret_cast<float*>(in->out);
    p.out_stride = (int)in->out_stride;

    const int slots = p.bsz * p.topk;
    const int nproj = p.gated ? 2 : 1;
    const int grid_a = slots * nproj * (p.I / 128);
    const int grid_b = slots * (p.Ho / 128);
    void* args[] = { &p };
    CUresult st;
    if (p.a_global) {
        const int rot_items = slots * (p.Hi / 128) * nproj;
        const int rot_grid = (rot_items + 15) / 16;
        st = cuLaunchKernel(mod->rot, rot_grid, 1, 1, 512, 1, 1, 0, stream, args, nullptr);
        if (st != CUDA_SUCCESS) return fail(st, "rot");
    }
    st = cuLaunchKernel(mod->a, grid_a, 1, 1, 512, 1, 1, mod->smem_a, stream, args, nullptr);
    if (st != CUDA_SUCCESS) return fail(st, "kernel-a");
    st = cuLaunchKernel(mod->b, grid_b, 1, 1, 512, 1, 1, mod->smem_b, stream, args, nullptr);
    if (st != CUDA_SUCCESS) return fail(st, "kernel-b");
    return 0;
}
