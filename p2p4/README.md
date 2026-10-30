# Serving transport variants

Use the [transport v3 guide](TRANSPORT_V4.md) for installation, qualification and the complete answers to the HOSTPAIR report. This supersedes the original P2P4-only instructions.

```bash
python3 install_p2p4.py --transport both
export EXL3_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
export FP8_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-FP8
export ENGRAM_DIR=/absolute/path/to/DeepSeek-V4.1-Flash-1017-engram
export DSV41_GPUS=device=0,1,3,4
export DSV41_RANK_NODES="0 0 0 0"
DSV41_TRANSPORT=hostpair bash p2p4/run.sh
```

For native-payload full-group P2P use `DSV41_TRANSPORT=p2p4 DSV41_P2P_PAYLOAD_DTYPE=native`. Only run one candidate in the allocated GPU job. Use the correct rank/arena NUMA placement for the host. The launcher checks image transport/revision labels and selects the corresponding transport-v4 tag.

Run `docker exec CONTAINER python3 /opt/dsv41/p2p4-source/verify_optimized.py` after startup. Require four committed optimized mixed-bank receipts and generated verification tokens before timing. Standalone operator qualification is documented in TRANSPORT_V4.md and does not replace full-model admission or model-quality qualification.

The old `pair-test-result.json` and `startupfix-pair-result.json` are historical development checks for the preceding FP32-payload candidate; consult the transport-v4 build receipt and its referenced new evidence for this revision.
