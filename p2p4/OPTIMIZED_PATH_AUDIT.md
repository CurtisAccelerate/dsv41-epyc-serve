# Optimized-path audit

This audit follows executable selection through actual vLLM communicator construction, captured profile updates, R2 startup parity and serving admission. Full-model startup of the new variants on recipient EPYC is still required; no finite static/operator audit can prove that no host-specific blocker remains.

## Proven integration

- The actual platform factory returns Sm80CudaCommunicator. With the exact disabled-custom-AR CLI setting, four GPU processes instantiate it and select HOSTSHM/PYNCCL in that order. The fixed pair state initializes with the pinned original cubin. This tests the factory path that the historical STEER_3 donor failed to activate.
- The actual collective prepare/apply/commit implementation inventories real retained six- and 48-row graphs, patches the six profile and replays them correctly. This is an isolated synthetic graph fixture, not a model-loading claim.
- Every configured R2 batch (2, 3, 4, 5, 6, 8; batch 7 pads to 8) passes the same three-input bitwise planned-versus-incumbent check as R2 startup on four ranks. The large-message actual communicator fallback also satisfies its native dtype numerical contract.
- The strict receipt auditor accepts the original live Broadwell startup receipts: all four ranks have the mixed bank, all R2 components, B7 padding and all samplers. Unit mutations of readiness, exclusions, sampler state and contract identity are rejected. This was a read-only check; the live server was not restarted or recaptured.
- The host-network HTTP test returns 200 on an isolated Docker daemon configured without a working bridge. Launcher tests confirm host and bridge argv, localhost binding and port selection. Port conflicts are checked before model loading.
- The Engram placement observer now reads one byte per mapped page before querying numa_maps. A CPU fixture reproduces the old rejection after dropping PTEs, confirms the read-only prefault restores complete evidence, verifies unchanged payload bytes and still rejects a wrong node. This hardens the observation mechanism; it does not prove that mechanism caused every historical page-count shortfall or waive genuine NUMA failures.

## What counts as the optimized path

The source control now requires the complete optimized receipt check before attempting production prepare RPCs: all R2 components and samplers must be prepared without exclusions. It opens ordinary serving only after the required mixed arm commits on all four ranks. CPU fixtures execute the actual control method and confirm that incomplete receipts are rejected before RPCs and complete receipts can commit and resume. The additional `verify_optimized.py` repeats the full audit and requires generated-token serving verification. Readiness alone does not prove that every request hits every fast component: shape-specific serving proof still uses raw CUPTI phase/kernel traces and the replay ledger.

The model recipe retains runner 0, DSpark K5, TP4/PP1, native MLA KV, TC MoE/prefill, cooperative expert kernels, the attention and sampler banks, R2 batch planning and registered resident Engram backing. HOSTSHM/pair continues to use BF16 payloads with FP32 arithmetic. Full P2P uses ordinary NCCL with native payloads by default, not GNmac's cut-through ring. Pair execution remains restricted to its qualified 48-row size; large prefill uses the upstream fallback.

## Limits and required recipient checks

The build host has two peer pairs, not four-way direct connectivity. The new full-model EPYC run cannot be performed here while preserving the live Broadwell service. Direct two-rank P2P tests and separately labeled four-rank NCCL fallback diagnostics cover the available hardware; neither is presented as four-way direct P2P qualification.

Install both variants, require the CPU/network checks, run each four-GPU operator qualification in the recipient's exclusive wrapper, then start the model and require `verify_optimized.py` before benchmarking. Preserve the saved all-rank diagnostics on any rejection. Numerical correctness within a declared reduction bound is separate from full-model quality and measured performance. No speedup or 99% end-to-end success probability is claimed.
