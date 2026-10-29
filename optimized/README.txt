This package starts the optimized DeepSeek V4.1 Flash 1017 server.
It is a serving image, not the earlier report image. Do not overlay files from
dsv41-epyc-serve onto it and do not substitute the Morrow donor runtime.

LOAD

Unzip the kit beside the Docker archive, then run:

  sha256sum -c SHA256SUMS
  docker load -i dsv41-optimized-20261029-p2p.docker.tar.gz
  docker run --rm --network none dsv41-optimized:20261029-p2p check

RUN ON THE EPYC HOST

Use the matching original FP8 checkpoint, EXL3 checkpoint and Engram shards.
The exact hybrid config, mixed-bit layer selection and index are already inside
the image. No runtime source, TC hook, .so, cubin or SM80 mount is needed.

  export EXL3_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
  export FP8_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-FP8
  export ENGRAM_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-engram
  bash run.sh

ENGRAM_DIR may equal FP8_DIR if that directory contains the matching original
Engram shards 43, 47 and 48. Keep the exact source files consistent across runs.

run.sh selects exactly four CMP 170HX cards automatically, excluding another GPU
model. Override only if needed, for example:

  export DSV41_GPUS=device=0,1,3,4

Default NUMA placement is one node: DSV41_RANK_NODES='0 0 0 0'. Set this to the
four rank memory nodes appropriate for the host if it exposes several NUMA nodes.
The builder verifies actual placement and available RAM; it will not silently
copy the Engram tables onto a different storage tier.

The first start builds approximately 189 GiB of exact shared Engram backing in
tmpfs. It requires that payload plus RAM headroom, sufficient per-node capacity,
and unlimited memlock. run.sh sets tmpfs, shared memory and memlock limits and
adds SYS_NICE for NUMA policy calls. This image has no NVMe streaming fallback.
Wait for Docker health to become healthy; first load and graph qualification
can take substantial time.

VERIFY THE PATH ACTUALLY SERVING

  docker logs dsv41-optimized
  docker exec dsv41-optimized python3 /opt/dsv41/entrypoint.py verify-serving

Required startup evidence:
  DSV41_RUNTIME_CHECK: passed
  DSV41_P2P_CAPABILITY: all pairs of the four GPUs permit peer access
  DSV41_P2P_VERIFIED: NCCL all-reduce correct and P2P transport observed
  Docker health: healthy

verify-serving requires four successful commits of the optimized
next-attention-sampler-native-mixed graph bank, open public admission, and a
generated API response. A responsive /v1/models endpoint alone is insufficient.
Receipts live under /w9/state/dsv41-*/ inside this container and are created anew.

DEFAULT SERVING SETTINGS

TP4, PP1, DSpark K5, VLLM_USE_V2_MODEL_RUNNER=0 (the current stack's runner),
fp8_ds_mla, 262144 context, 8 sequences, 2048 batched tokens, async scheduling,
prefix caching, and captures 1,2,3,4,6,8,12,18,24,30,36,48.
DSV41_R2, attention bank/split, sampler bank, coop, TC MoE and TC prefill are on.
NCCL_P2P_DISABLE=0 and NCCL_P2P_LEVEL=SYS. All-pairs P2P is required by default.
The graph bank retains its optimized HOSTSHM and pair collectives: enabling
NCCL P2P does not replace every custom collective with NCCL.

SOURCE AND REBUILD

The image contains installed runtime Python, SM80 code, all required custom
modules, exact compiled serving kernels, and optimization CUDA sources.
/opt/dsv41-build contains the ExLlama/plugin sources; /opt/dsv41/source contains
the gate/coop and TC sources. Package versions are in packages.json.
The source archive includes the custom runtime and optimization sources.

To compile the K2/K3 gate libraries without touching the serving pins:

  docker run --rm --entrypoint bash dsv41-optimized:20261029-p2p \
    /opt/dsv41/rebuild-gates.sh

Rebuilt bytes require correctness/performance qualification before deployment.
The shipped serving libraries remain the captured current build's exact bytes.

Dockerfile reconstructs the image from its exported sanitized root filesystem:
create a stopped container, docker export it to runtime-rootfs.tar, then build
with the included Dockerfile. Upstream compiled dependencies are shipped and
version-inventoried; this is not a clean upstream vLLM C++ rebuild recipe.

VALIDATION SCOPE

See build-receipt.json and privacy-audit.json for actual results and image ID.
Source hashes, dependency imports, GPU-dependent imports and NUMA portability
checks passed. K2/K3 gate sources compiled with the included headers/toolchain.
Full model inference and performance on the new image remain untested: the live
server occupies the four GPUs. The build did not restart or reconfigure it.
The build host only permits P2P within two pairs; strict all-pairs checking
correctly refuses that topology. The recipient must pass its own P2P check.
Historical comparison measurements are in matched-benchmarks.json.

Credentials, live state, benchmark tokens, machine identifiers, cache contents,
Git history, weights and Engram payloads are excluded. The final filesystem is
flattened before distribution. Public crypto parser markers, distribution
crypto fixtures and SDK examples are identified separately in the audit.
