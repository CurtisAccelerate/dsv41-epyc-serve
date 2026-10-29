# Install the complete pinned image first: python3 install_image.py
ARG BASE_IMAGE=dsv41-optimized:20261029-p2p
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/CurtisAccelerate/dsv41-epyc-serve"
ENTRYPOINT ["python3", "/opt/dsv41/entrypoint.py"]
CMD ["serve"]
