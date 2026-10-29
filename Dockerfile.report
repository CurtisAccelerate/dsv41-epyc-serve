# Source image for the October 28 serving selection.
# The default base is small and builds anywhere. Pass a CUDA devel image when
# the recipient is going to compile the kernel sources inside the image:
#   docker build --build-arg BASE=nvidia/cuda:12.8.1-devel-ubuntu24.04 .
ARG BASE=python:3.12-slim-bookworm
FROM ${BASE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends pciutils \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/dsv41-serve
ENV DSV41_P2P=all \
    NCCL_P2P_DISABLE=0 \
    NCCL_P2P_LEVEL=SYS \
    DSV41_R2=1 \
    DSV41_R2_PAD7=1 \
    DSV41_R2_BATCHES=2-8 \
    DSV41_R2_CAP_MIB=1536 \
    DSV41_R2_TRACE_BUDGET=1:4,2:4,3:4,4:4,5:4,6:4,7:4,8:4 \
    DSV41_R1_FEATURES=1 \
    DSV41_DSPARK_V1=1 \
    W26_ATTENTION_BANK=1 \
    W26_ATTENTION_SPLIT=1 \
    W26_SAMPLER_BANK=1 \
    W14_COLLECTIVE_PROFILES=1 \
    W15_PAIR_TRANSPORT=1 \
    W9_EXL_AB_ENABLE=1 \
    W9_EXL_AB_OFFLINE_EXCLUSIVE=1 \
    W9_EXL_AB_EXTRA_GRAPH_BYTES=536870912 \
    W9_ENGRAM_ALLOW_MUTABLE_SHARED=1 \
    W9_RESIDENT_QUARANTINE=1 \
    VLLM_EXL3_COOP=1 \
    VLLM_EXL3_MOE_TP_ALIGN=128 \
    VLLM_EXL3_MOE_TP_ROTATE=1 \
    VLLM_EXL3_TC_MOE_V2=1 \
    VLLM_EXL3_TC_MOE_PREFILL=1 \
    VLLM_EXL3_TC_PREFILL_ROWS=2048 \
    VLLM_EXL3_ARENA_CACHE=0 \
    W26_DSPARK_CAPTURE_LOCAL=0 \
    W26_DSPARK_LOCAL=0 \
    W26_MHC_DUAL_GRAPH=0 \
    W26_PREFILL_SPLIT_REDUCE=0 \
    W26_ENGRAM_EARLY=0 \
    W26_OBSERVE=0 \
    W26_OBSERVE_ROUTES=0 \
    W26_OBSERVE_MARGINS=0
RUN chmod 0755 /opt/dsv41-serve/entrypoint.sh /opt/dsv41-serve/src/kernels/gu-fp32/build.sh \
    && python3 /opt/dsv41-serve/share-check.py /opt/dsv41-serve \
    && python3 /opt/dsv41-serve/p2p/policy.py \
    && python3 /opt/dsv41-serve/report/report.py --selftest

WORKDIR /opt/dsv41-serve
ENTRYPOINT ["/opt/dsv41-serve/entrypoint.sh"]
CMD ["report"]
