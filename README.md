# Complete optimized DeepSeek V4.1 Flash serving image

## EPYC: four-way GPU P2P

Use the new [EPYC P2P4 build](p2p4/README.md) for full four-rank GPU-memory TP all-reduce. It retains the optimized serving path and disables HOSTSHM and paired transport. From this checkout:

```bash
python3 install_p2p4.py
```

Then run the four-GPU qualification and `bash p2p4/run.sh` as documented in the [EPYC guide](p2p4/README.md). The installed image is `dsv41-optimized:20261029-epyc-p2p4`. This installs the verified complete base image and builds a small, pinned transport layer locally. The [EPYC release kit](https://github.com/CurtisAccelerate/dsv41-epyc-serve/releases/tag/epyc-p2p4-20261029) contains the same installer and source.

The image has been built and passed source/GPU-import checks, transport-plan checks, and two-rank eager/captured-graph operator checks. **Full four-GPU serving, quality and performance remain pending on EPYC.** Missing peer links are rejected. No speedup is promised before a matched benchmark. See the [build receipt](optimized/p2p4-build-receipt.json).

## Original hybrid base and control

The following instructions install the previous image with HOSTSHM and paired collectives still selected in its graph bank. It remains the complete pinned base for the new transport and an A/B control. For the requested EPYC four-way path, use `install_p2p4.py` and `p2p4/run.sh` above.

The complete sanitized image is distributed through the [optimized-20261029-p2p release](https://github.com/CurtisAccelerate/dsv41-epyc-serve/releases/tag/optimized-20261029-p2p), with its source kit and installed-file manifests. The approximately 9.3 GB Docker archive is split into five release assets. The installer downloads each part, checks each SHA256 and the combined archive, loads it, verifies the exact image ID, and checks installed sources and imports.

## Install

Requires Python 3, Docker, and at least 50 GB of available Docker/download storage. From this checkout:

```bash
python3 install_image.py
unzip dsv41-download/dsv41-optimized-kit.zip -d serving-kit
```

Use `sudo python3 install_image.py` if your account cannot access Docker. Downloads can be resumed by rerunning the installer; complete verified parts are reused. `--download-only` downloads and verifies without loading or starting a container.

## Serve

```bash
export EXL3_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
export FP8_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-FP8
export ENGRAM_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-engram
bash serving-kit/run.sh
```

The launcher selects four CMP 170HX cards automatically. Override with `DSV41_GPUS=device=0,1,3,4` if necessary. Only checkpoint data is mounted; the runtime, SM80 code, worker modules, Engram loader, TC hook and compiled optimized libraries are installed inside the image. The exact hybrid model config and index are included. The Engram directory may equal FP8_DIR if it contains the matching original shards.

Default NUMA placement is `DSV41_RANK_NODES="0 0 0 0"`; adjust for hosts exposing several NUMA nodes. Startup prepares approximately 189 GiB of shared resident Engram backing and enforces available-RAM and placement checks. There is no automatic NVMe streaming fallback.

## Optimized defaults and proof

TP4, DSpark K5, `VLLM_USE_V2_MODEL_RUNNER=0`, `DSV41_R2=1`, attention bank/split, sampler bank, coop, tensor-core MoE and prefill, `fp8_ds_mla`, and captures `1 2 3 4 6 8 12 18 24 30 36 48`. P2P is enabled with `NCCL_P2P_DISABLE=0` and `NCCL_P2P_LEVEL=SYS`. The graph bank retains its custom HOSTSHM and pair collectives for their selected shapes.

Startup requires peer capability for every GPU pair and a correct NCCL all-reduce with direct P2P logged on every rank, without SHM/NET data fallback. Serving admission opens after graph-bank qualification. Verify with:

```bash
docker exec dsv41-optimized python3 /opt/dsv41/entrypoint.py verify-serving
```

This requires four committed optimized-bank receipts, open admission and a generated API response. A models endpoint or flags alone is insufficient.

Pinned image ID: `sha256:e3bc13c775dee4013f5cfe28688482e0ec8ccb9b045ccb012c9f6e53b56c2726`.

Checks passed: installed-file identities, CPU/GPU imports, K2/K3 source compilation, NUMA portability, privacy scan, and direct NCCL P2P on two connected GPUs. Full four-GPU inference and performance on the new image remain pending on the recipient host. The build host only permits P2P within two pairs; its strict all-pairs test correctly refuses that topology. See [build receipt](optimized/build-receipt.json) and [privacy audit](optimized/privacy-audit.json).

## Source and existing files

The release's source kit contains 6,462 source/configuration files, CUDA headers and rebuild tools. Exact compiled serving libraries are included in the image. Upstream compiled dependencies are shipped and version-inventoried; a clean upstream vLLM C++ build recipe is outside this snapshot.

After installing P2P4 with `install_p2p4.py`, `docker build -t dsv41-epyc-serve:optimized .` adds this repository's source label to that complete image. `Dockerfile.report` and `README-report.md` preserve the historical report-only package. Existing `src/`, report and configuration files are historical references; use the release snapshot for the serving runtime.
