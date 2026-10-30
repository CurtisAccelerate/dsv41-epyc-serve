# Transport v4: HOSTSHM/pair and native-payload P2P

This revision resolves the configuration barriers in CURTIS_HOSTPAIR_REPORT. It supplies two complete serving variants from the same pinned base, with matching runner 0, expert/attention/sampler banks and model recipe. It does not change the running Broadwell service or implement GNmac's cut-through ring.

The deeper verification also fixes the recipient's known bridge-network problem: the launcher defaults to host networking with a localhost API, checks the selected port before model loading, and the installer runs a CPU-only HTTP smoke test. `DSV41_NETWORK=bridge` is available only for hosts with a functioning Docker bridge. No daemon/network configuration change is required.

## Install and select

```bash
python3 install_p2p4.py --transport both
```

This installs `dsv41-optimized:20261030-transport-v4-hostpair` and `dsv41-optimized:20261030-transport-v4-p2p4`. The original complete image download is reused when installed. Build dependencies are already in the complete base. No old derivative image or single-file transplant is needed. HOSTSHM builds its inline CUDA extension on first initialization; allow several minutes and retain the compiler log if it fails.

Set the existing EXL3_DIR, FP8_DIR and ENGRAM_DIR. Run GPU commands only inside the recipient's canonical exclusive wrapper. Keep the same four CMPs, power, clocks, prompts and runner for comparisons. Set DSV41_RANK_NODES to the host's actual placement; the launcher derives the two arena nodes from ranks 0 and 2.

```bash
# A: HOSTSHM plus the fixed 48-row pair kernel
DSV41_TRANSPORT=hostpair DSV41_CONTAINER=curtis-hostpair bash p2p4/run.sh

# B: full-group NCCL with input dtype preserved; BF16 tensors stay BF16
DSV41_TRANSPORT=p2p4 DSV41_P2P_PAYLOAD_DTYPE=native \
  DSV41_CONTAINER=curtis-p2p-native bash p2p4/run.sh

# C: optional FP32-widening comparison, explicitly requested
DSV41_TRANSPORT=p2p4 DSV41_P2P_PAYLOAD_DTYPE=fp32 \
  DSV41_CONTAINER=curtis-p2p-fp32 bash p2p4/run.sh
```

Run one candidate at a time; container names do not reserve GPUs. Each image rejects contradictory transport flags before model preparation. The launcher checks both transport and transport-v4 revision labels. Payload policy is fixed when the communicator initializes; changing it in place is rejected. Native NCCL changes the arithmetic/reduction contract relative to FP32 payloads; it is not a bitwise-equivalence or model-quality claim.

## Operator qualification before model startup

Use the same four GPUs as serving; replace the sample indices with the host's actual four CMP indices. These commands do not load model weights.

```bash
docker run --name hostpair-proof --gpus '"device=0,1,3,4"' \
  --cap-add SYS_NICE --shm-size 2g --ulimit memlock=-1:-1 \
  -e DSV41_ARENA_NODES='0 0' --entrypoint python3 \
  dsv41-optimized:20261030-transport-v4-hostpair \
  /opt/dsv41/p2p4-source/qualify_hostpair.py
docker cp hostpair-proof:/tmp/dsv41-hostpair-qualification ./hostpair-proof

docker run --name p2p-native-proof --gpus '"device=0,1,3,4"' --shm-size 2g \
  -e DSV41_P2P_PAYLOAD_DTYPE=native --entrypoint python3 \
  dsv41-optimized:20261030-transport-v4-p2p4 \
  /opt/dsv41/p2p4-source/qualify_transport.py
docker cp p2p-native-proof:/tmp/dsv41-p2p4-qualification ./p2p-native-proof
```

Set arena nodes correctly for the machine. If the container's default seccomp policy blocks mbind/move_pages, use the existing approved NUMA-capable GPU wrapper policy; do not relax placement checks. Repeat the P2P command with `fp32` and a new container/output name for that comparison. Preserve all receipts, then remove only the completed owned proof containers. Require all four rank receipts and independent-reference/graph checks. The P2P result must say `full_four_way_qualified=true` on EPYC; two-rank development results do not qualify it.

After full model startup, run `docker exec CONTAINER python3 /opt/dsv41/p2p4-source/verify_optimized.py`. For a nondefault API port, append `--url http://127.0.0.1:PORT`. This runs the ordinary serving verifier and additionally requires all four initial mixed-arm receipts, all B2–B8 R2 graph components with no exclusions, qualified B7 padding and all greedy/sampled samplers. It rejects partial optimized preparation. Require generated API tokens and transport-specific readiness; operator tests do not substitute for this gate.

The native-payload numerical check compares with CPU FP64 using the input-dependent `gamma_(world_size-1) * sum(abs(inputs))` reduction-roundoff bound, with underflow and final-cast terms. A fixed relative tolerance against the final BF16-rounded CPU sum can incorrectly reject valid four-rank BF16 reductions near cancellation. The new contract explicitly permits the native dtype's intermediate rounding; it does not establish original-engine equivalence or full-model quality. FP32-wire mode retains its explicit final-output rounding contract. Tests reject grossly wrong and nonfinite results and check bitwise own-graph and restored-input repeatability. The optional `--allow-host-fallback-test` is an arithmetic diagnostic only and always emits `full_four_way_qualified=false`; it must not be used as EPYC P2P admission evidence.

## Answers to the four HOSTPAIR questions

1. **Topology:** HOSTSHM/pair requires directed partner links 0↔1 and 2↔3. Extra peer capabilities are accepted; the kernel still opens only the rank-XOR-1 IPC buffer. Full-group P2P requires every directed link. Both startup preflight and pair initialization enforce the same policy. CPU tests enumerate all 256 optional cross-pair edge combinations and reject every missing mandatory partner edge. The cubin and schedule are unchanged.
2. **Scope:** pair remains exactly 491,520 BF16 bytes, hidden 5120 × 48 rows, eight blocks × 256 threads. R2 selects it for B8 and padded B7. B1 and other eligible shapes use HOSTSHM. HOSTSHM's default cap is 512 KiB; a 2048-row/20-MiB prefill residual goes to the upstream NCCL fallback. No new large-prefill pair kernel is claimed. TC expert prefill acceleration is separate from residual collectives.
3. **Execution proof:** the classifier now counts `pair_write_bf16` as a collective. Set DSV41_R1_TRACE_MODE=cupti, DSV41_R1_FEATURES=1 and an owned DSV41_R1_OUT. Use otherwise identical requests with `vllm_xargs: {"dsv41_trace": 2}` in the request body. This is supported extra request metadata in this stack, not an OpenAI-standard field. Retain every rank's raw kernel names, phase/round records and graph-node IDs; require successful CUPTI self-test, cupti mode, zero dropped activity/sink records and complete rank coverage. `pair_write_bf16` is expected only when the 48-row route actually executes. The standalone host/pair proof inspects graph kernel names, waits for execution, and checks device-completed generations and sticky errors. Python enqueue counters alone are not replay proof. Capture paused/drained status before and after isolated serving requests to attribute generation deltas; use raw prefill phase records rather than decode-only receipts for prefill claims.
4. **Payload dtype:** `DSV41_P2P_PAYLOAD_DTYPE=native` is now the default and preserves BF16/FP16 payload dtype; it avoids this wrapper's FP32 tensor copy and cast-back. `fp32` reproduces the widening policy for comparison. Startup receipts expose the chosen policy and graph provenance records actual wire bytes/dtype. This controls ordinary NCCL, not GNmac's custom ring. Full-model numerical/quality tests remain necessary.

The historical EPYC STEER_3 CUSTOM/PYNCCL result is not a measured HOSTSHM/pair baseline. New matched A/B runs are needed. Do not compare GNmac 180 W headline rates directly with these 140 W runs or infer a speedup from operator correctness alone. The release build receipt states the exact operator tests performed and remaining EPYC/full-model limits.
