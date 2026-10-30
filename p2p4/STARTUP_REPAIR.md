# Consolidated EPYC startup repair — 2026-10-30

This candidate rebuilds from the original pinned complete base. It includes the preflight interpreter fix, stale control-source pin repair and transport-aware application-state journal. It also replaces raw-handle capture bookkeeping with live-owner/capture-generation provenance and saves all rank prepare/apply receipts before comparing their counts.

The four-way P2P transport was demonstrated on EPYC in the recipient's report. The latest EPYC failure was a collective inventory mismatch before the attention census ran. Raw-handle reuse is a demonstrated bug in the prior registry, but the report did not retain the failed count maps, so its role in that particular failure remains unproved.

The new code passes 96 real CUDA reset/recapture cycles per rank, including actual raw-handle reuse, the exact vLLM segmented capture wrapper, and the existing 36 operator/graph cases per rank on two connected GPUs. Assembly checks verify launch configuration, manifests, legacy/P2P census branching and rejection diagnostics. Full optimized four-GPU model startup on EPYC is still required; these checks do not establish 99% confidence in successful admission.

## Recipient procedure

1. Use this release's complete kit. Run `python3 install_p2p4.py`; it reuses the original complete base if present and builds `dsv41-optimized:20261030-epyc-p2p4`. Do not reuse the previous variant tag or copy only one patched module into an old image. The installer checks 4,576 source/metadata references, interpreter launch and rejection diagnostics. It supports Docker stores reporting an OCI manifest ID by checking the exact rootfs layer IDs and runtime configuration.
2. Keep the existing exclusive GPU wrapper, four selected CMP GPUs, model mounts, 140 W policy and runner 0. Within that scheduled allocation, repeat `qualify_transport.py` on all four GPUs. Require `full_four_way_qualified=true`; the included two-rank evidence is development validation only.
3. Launch through `p2p4/run.sh` with the existing model paths and a distinct `DSV41_CONTAINER` name. Require attention, R2 and `next-attention-sampler-native-mixed` readiness, then run `entrypoint.py verify-serving` in that container before any benchmark.
4. Preserve the complete container log and `/w9/state` before removing a failed container. Collect `collective-<token>.json` from its state directory: these now contain every rank's counts and per-graph semantic inventory even when the comparison fails. Also retain attention-initial, serving status, optimized-serving receipts when present, and NCCL logs.
5. If another package-local defect appears, continue the bounded diagnosis/repair/requalification loop within the allocated job, preserving exact changes and updated manifests. Keep numerical, state restoration, NUMA, graph coverage, all-rank agreement and admission checks enforced. Do not change host software or bypass the canonical GPU wrapper. If a fix cannot be validated within the allocation, return the complete diagnostics rather than only a headline exception.

## Gates retained

Legacy HOSTSHM still requires explicit communication exclusions. P2P mode requires a live TP4 owner and stable communicator/device/NCCL-handle binding with no host registry. The journal never restores NCCL protocol state. Captures without current sealed owner provenance fail. Every accepted graph requires executable NCCL nodes and payload metadata. All-rank count equality remains mandatory; a fixture confirms a genuine mismatch still quarantines and now leaves its rank maps on disk.
