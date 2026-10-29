# EPYC four-way P2P serving

This builds the complete optimized serving runtime with a full four-rank GPU-memory NCCL all-reduce. HOSTSHM and pair transport are disabled. The pinned expert, attention, DSpark K5, sampler, TC MoE/prefill, capture plan and resident Engram optimizations remain installed.

The implementation has passed source/dependency checks and two-rank eager and CUDA graph operator checks. **Four-GPU inference, quality, acceptance and performance are pending on EPYC.** The build host supports peer access only within two pairs, so it cannot qualify the four-way mode. No measured speedup is claimed.

## Install

From the repository root (or unpacked EPYC release kit):

```bash
python3 install_p2p4.py
```

Requires Linux, Python 3, Docker, NVIDIA Container Toolkit, four peer-accessible SM80 GPUs and sufficient host memory for the approximately 189 GiB resident Engram backing. The installer reuses the exact complete base image if installed; otherwise it downloads the verified 9.3 GB base from GitHub Releases. Allow at least 50 GB of free Docker/download storage. It then builds and verifies `dsv41-optimized:20261029-epyc-p2p4` locally, without fetching build dependencies. Model weights are supplied separately.

## Qualify four-way transport

Select exactly the four intended GPUs; the example indices must be adjusted to the host. This checks operator correctness and captured graphs, without loading the model:

```bash
docker run --name dsv41-p2p4-qualification --gpus '"device=0,1,3,4"' \
  --shm-size 2g --entrypoint python3 dsv41-optimized:20261029-epyc-p2p4 \
  /opt/dsv41/p2p4-source/qualify_transport.py
docker cp dsv41-p2p4-qualification:/tmp/dsv41-p2p4-qualification ./p2p4-qualification
docker rm dsv41-p2p4-qualification
```

Require `passed: true`, `world_size: 4`, `full_four_way_qualified: true` and all four rank receipts. The test exercises BF16, FP16 and FP32 reductions at ten sizes, compares with an independent CPU FP64 reference, and replays captured graphs with changed inputs and poisoned outputs. Every rank must log direct P2P without SHM/NET data fallback. Missing peer access aborts. The included `pair-test-result.json` is explicitly a two-rank development check and does not qualify four-way serving.

## Serve and verify

```bash
export EXL3_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
export FP8_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-FP8
export ENGRAM_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-engram
export DSV41_GPUS=device=0,1,3,4
export DSV41_RANK_NODES="0 0 0 0"
bash p2p4/run.sh
```

The launcher auto-selects four CMP 170HX cards if `DSV41_GPUS` is omitted. Set NUMA nodes to match the host. The default container name is `dsv41-optimized`; choose `DSV41_CONTAINER` for an additional instance. The default port is 8095. Check the running instance in another terminal:

```bash
docker exec dsv41-optimized python3 /opt/dsv41/entrypoint.py verify-serving
```

Verification requires four committed optimized-bank receipts, four TP4 GPU transport receipts, open admission and generated API tokens. Graph receipts require captured NCCL kernel nodes and recorded residual all-reduces. CPU/source checks and transport-only qualification do not establish full serving correctness.

## Transport and comparison

The image fixes `DSV41_TRANSPORT=p2p4`, `VLLM_HOST_ALLREDUCE=0`, `W15_PAIR_TRANSPORT=0`, `NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS` and `NCCL_SHM_DISABLE=1`, and disables vLLM's alternative custom all-reduce. Every TP all-reduce goes through the same four-rank PyNCCL communicator. Half inputs are promoted on GPU to FP32 for accumulation, then cast back on GPU. The changed reduction order requires model quality and speculative acceptance checks. Promotion adds bandwidth and launch costs.

The internal `six` controller label is retained for compatibility with the existing bank transaction; it does not choose a host kernel. Transport receipts explicitly identify `p2p4`, world size four and zero host-staged payload bytes. NCCL may use host memory for control/bootstrap; the assertion concerns tensor payload transport.

Compare the original hybrid image and this image on the same EPYC host, checkpoints, prompts, output lengths, K5 configuration, GPU power/clocks and warmup. Measure C1/C4/C8 output throughput, latency, acceptance, output correctness and GPU utilization. Keep qualification JSON, serving verification and benchmark results together. Morrow's gain is a comparison target, not a guaranteed outcome of enabling P2P.
