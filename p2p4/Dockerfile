ARG BASE_IMAGE=dsv41-optimized:20261029-p2p
FROM ${BASE_IMAGE}
ENV DSV41_TRANSPORT=p2p4 \
    VLLM_HOST_ALLREDUCE=0 \
    W15_PAIR_TRANSPORT=0 \
    NCCL_P2P_DISABLE=0 \
    NCCL_P2P_LEVEL=SYS \
    NCCL_SHM_DISABLE=1 \
    DSV41_REQUIRE_P2P=1 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P \
    NCCL_DEBUG_FILE=/tmp/dsv41-nccl4.%p.log
COPY . /opt/dsv41/p2p4-source/
RUN python3 /opt/dsv41/p2p4-source/install_overlay.py
LABEL org.opencontainers.image.source="https://github.com/CurtisAccelerate/dsv41-epyc-serve"
LABEL io.dsv41.transport="nccl-p2p4"
ENTRYPOINT ["python3", "/opt/dsv41/entrypoint.py"]
CMD ["serve"]
