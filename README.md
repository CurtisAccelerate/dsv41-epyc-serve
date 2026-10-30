# Complete optimized DeepSeek V4.1 Flash serving image

## Transport v4

The [transport v4 release](https://github.com/CurtisAccelerate/dsv41-epyc-serve/releases/tag/transport-v4-20261030) provides complete HOSTSHM/pair and full-group NCCL P2P variants from the same pinned base and serving recipe.

```bash
python3 install_p2p4.py --transport both
```

The installer reuses or downloads the verified complete base, builds the two small source layers locally and checks their installed manifests and startup contracts. Model weights are supplied separately. Requires Linux, Python 3, Docker, NVIDIA Container Toolkit and sufficient host memory for approximately 189 GiB of resident Engram backing. Allow at least 50 GB of available Docker/download storage.

Read the [transport guide and answers to the HOSTPAIR questions](p2p4/TRANSPORT_V4.md) before running. It covers model mounts, GPU/NUMA selection, standalone qualification, serving verification, tracing and matched A/B measurements.

- `DSV41_TRANSPORT=hostpair`: the required partner links 0↔1 and 2↔3 must exist; extra peer capability is accepted. Pair execution remains fixed at 48 rows, with HOSTSHM for other eligible sizes and upstream fallback for larger messages.
- `DSV41_TRANSPORT=p2p4`: every directed peer link must exist. `DSV41_P2P_PAYLOAD_DTYPE=native` preserves the input representation and is the default; `fp32` explicitly enables the previous widening policy for comparison. This is ordinary NCCL, not GNmac's custom ring.

Both retain runner 0, DSpark K5, optimized expert/attention/sampler banks and the resident Engram recipe. Neither configuration is a claim of exact original-model numerical equivalence or guaranteed speedup. Numerical, NUMA, graph, all-rank and serving admission gates remain enforced.

See the [build/test receipt](optimized/transport-v4-build-receipt.json) for actual qualification and limitations. Native-payload P2P and HOSTSHM/pair on fully connected EPYC still require recipient four-GPU/full-model validation before benchmarking. The original pair cubin is unchanged.

## Original image and history

The original complete base remains in the [optimized-20261029-p2p release](https://github.com/CurtisAccelerate/dsv41-epyc-serve/releases/tag/optimized-20261029-p2p). Its approximately 9.3 GB archive has five checksum-verified parts and a source kit. The transport-v4 installer uses it as an immutable dependency; do not use its old launcher as the HOSTSHM/pair EPYC control because its old topology checks conflict.

The earlier P2P4 candidate and startup repair releases are retained for reproducibility. `p2p4/STARTUP_REPAIR.md` documents that earlier revision. `Dockerfile.report` and `README-report.md` preserve the historical report-only package. Existing `src/` is historical reference; use the installed release snapshot for serving.
