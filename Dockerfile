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
    W26_ATTENTION_BANK=1 \
    W26_ATTENTION_SPLIT=1 \
    W26_SAMPLER_BANK=1 \
    W14_COLLECTIVE_PROFILES=1 \
    W15_PAIR_TRANSPORT=1 \
    W26_DSPARK_CAPTURE_LOCAL=0 \
    W26_MHC_DUAL_GRAPH=0
RUN chmod 0755 /opt/dsv41-serve/entrypoint.sh /opt/dsv41-serve/src/kernels/gu-fp32/build.sh \
    && python3 /opt/dsv41-serve/share-check.py /opt/dsv41-serve \
    && python3 /opt/dsv41-serve/p2p/policy.py \
    && python3 /opt/dsv41-serve/report/report.py --selftest

WORKDIR /opt/dsv41-serve
ENTRYPOINT ["/opt/dsv41-serve/entrypoint.sh"]
CMD ["report"]
