# dsv41-epyc-serve

Source package for an EPYC GPU-peer experiment. The default turns peer access
on for every card, which is the widest and fastest theoretical setting. The
October 28 measured stack is still here, as a separate config, so a run can
be compared with the published Morrow and Spark numbers.

This tree is meant to be the root of a repo. Clone it, build the image, run
the report. Weights, Engram tables, and a private runtime image are not
included.

The report prints the NCCL exports for the mode you selected. It does not
change a power cap, and it does not load a model. Put those exports on the
process that actually serves.

## Configure P2P

One variable, `DSV41_P2P`. Three files:

| file | value | what it does |
|---|---|---|
| `config/all.env` | `all` | **Default.** Peer every NVIDIA topo link the driver allows: `PIX`, `PXB`, `PHB`, `NODE`, `SYS`, and NVLink. Prints `NCCL_P2P_DISABLE=0` and `NCCL_P2P_LEVEL=SYS`. |
| `config/root-complex.env` | `root-complex` | Peer only inside one PCI host bridge (`PIX`, `PXB`, `PHB`). `NODE` and `SYS` stay off. Prints `NCCL_P2P_LEVEL=PHB`. |
| `config/measured.env` | `none` | The measured stack. No GPU peers. Host all-reduce. Prints `NCCL_P2P_DISABLE=1`. |

On one EPYC socket with each GPU on its own root port, topo between those GPUs is `NODE`. `all` is the mode that crosses those ports. `root-complex` enables nothing in that layout. `none` is how the published benches were measured.

Precedence, highest first:

1. `DSV41_P2P` already in the environment (`docker run -e` or `--env-file`)
2. `DSV41_P2P_FILE`, a mounted env file, only when `DSV41_P2P` is unset
3. `config/all.env`, loaded by the image when nothing else is set

Docker itself gives `-e` priority over `--env-file` when both are passed.

## Build and run

```
docker build -t dsv41-epyc-serve:source .
```

Fastest theoretical setting. This is the default, so the `-e` is optional:

```
docker run --rm --gpus all dsv41-epyc-serve:source
docker run --rm --gpus all --env-file config/all.env dsv41-epyc-serve:source
```

Same host bridge only:

```
docker run --rm --gpus all --env-file config/root-complex.env dsv41-epyc-serve:source
```

Measured stack, for comparison with the published numbers:

```
docker run --rm --gpus all --env-file config/measured.env dsv41-epyc-serve:source
```

Any of those can also be spelled `-e DSV41_P2P=all`, `-e DSV41_P2P=root-complex`, or `-e DSV41_P2P=none`.

`docker run --rm dsv41-epyc-serve:source check` reruns the share check and the selftests.

## What the report prints

CPU, each GPU, the topo matrix, and enable or leave-off for every pair. Then the measured settings (tensor parallel 4, 8 sequences, 2048 batched tokens, 262144 context, `fp8_ds_mla`, draft depth 5, the graph plan). Then the matched benches. The last line is one JSON object with the active mode and the measured card together.

Copy the `export` lines from `this run` onto the server you are timing. A `measured export` line is the comparison setting, not the active one, unless this run is `none`.

## Measured settings

These produced the benches. They are not the default of this image.

- P2P `none`, host all-reduce, `NCCL_P2P_DISABLE=1`
- Tensor parallel 4, max sequences 8, max batched tokens 2048, max length 262144
- KV cache `fp8_ds_mla`, draft depth 5, temperature 0, thinking off
- CUDA graph mode `FULL_AND_PIECEWISE`
- Capture sizes `1 2 3 4 6 8 12 18 24 30 36 48`
- 6 rows per user. 7 users are padded onto the 48-row graph
- 1 user: previous mixed-attention kernel
- 2–6 users: rebuilt gate/up/down
- 7–8 users: 48-row tensor-core experts
- Comparison host: 4× CMP 170HX at 140 W

Morrow serving tok/s (decode, median of 3, 400 output tokens): users 1–8 count `205.7 153.9 138.7 120.2 108.2 98.5 74.0 81.6`. Spark suite aggregate on that same host, not a DGX Spark: users 1, 2, 3, 4, 5, 6, 8 are `96.2 154.4 197.0 209.3 231.2 243.8 275.7`. The 1-user Spark row was a fused trial and is not the serving path. Full tables are in `benchmarks/matched.json`. These numbers describe that host. They are the comparison card, not a claim about the machine running the image.

## Source in this tree

- `src/kernels` — EXL3 coop sources and the gate/up/down launchers
- `src/kernels/gu-fp32/build.sh` — nvcc lines for those launchers
- `src/hybrid` — serving selection, graph capture, and native launch Python
- `src/patched` — runtime patches used with that selection, including host all-reduce
- `src/model/config.json` — model shape only
- `p2p/policy.py` — the three modes
- `report/report.py` — the report
- `benchmarks/matched.json` — Morrow and Spark cards
- `graphs/capture-plan.json` — which row count uses which kernel

`launch.cu`, `joint_launch.cu`, and `joint_kernels.cu` include `exllamav3_ext/quant/exl3_moe_coop_kernel.cuh`. That header is not in this repo. Point `EXL3_INCLUDE` at an ExLlamaV3 tree that contains it, then run `build.sh`. For that compile, build on a CUDA devel image:

```
docker build --build-arg BASE=nvidia/cuda:12.8.1-devel-ubuntu24.04 -t dsv41-epyc-serve:source .
```

`exl3_moe_coop.cu` needs the rest of that extension (ATen and the headers it includes). `build.sh` does not compile it alone. Without `EXL3_INCLUDE`, `build.sh` exits 1 and says so.

The loader reads an optional safetensors index from `DSV41_SELECTED_INDEX`. The arena cache directory defaults to the relative path `exl3-arena-cache`; override it with `VLLM_EXL3_ARENA_CACHE_DIR`. Both stay empty of weights until you mount them.

## Sanitization

The image build runs `share-check.py` and fails if the tree contains a private IPv4 address, a home directory, a `/mnt` path, a Windows profile path, a GPU UUID, a cloud key id, or a private key. Do not add host names, disk mounts, or weight files to this repo. The check does not know your machine; it only knows those patterns.
