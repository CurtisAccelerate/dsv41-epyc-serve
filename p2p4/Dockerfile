ARG BASE_IMAGE=dsv41-optimized:20261029-p2p
FROM ${BASE_IMAGE} AS source
COPY . /opt/dsv41/p2p4-source/
LABEL org.opencontainers.image.source="https://github.com/CurtisAccelerate/dsv41-epyc-serve"
LABEL io.dsv41.startup-contract="transport-v4"
ENTRYPOINT ["python3", "/opt/dsv41/entrypoint.py"]
CMD ["serve"]

FROM source AS hostpair
ENV DSV41_TRANSPORT=hostpair VLLM_HOST_ALLREDUCE=1 W15_PAIR_TRANSPORT=1 NCCL_SHM_DISABLE=0 \
    NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS DSV41_REQUIRE_P2P=1
RUN python3 /opt/dsv41/p2p4-source/install_overlay.py
LABEL io.dsv41.transport="hostpair"

FROM source AS p2p4
ENV DSV41_TRANSPORT=p2p4 VLLM_HOST_ALLREDUCE=0 W15_PAIR_TRANSPORT=0 NCCL_SHM_DISABLE=1 \
    NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS DSV41_REQUIRE_P2P=1 DSV41_P2P_PAYLOAD_DTYPE=native \
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P NCCL_DEBUG_FILE=/tmp/dsv41-nccl4.%p.log
RUN python3 /opt/dsv41/p2p4-source/install_overlay.py
LABEL io.dsv41.transport="nccl-p2p4"
