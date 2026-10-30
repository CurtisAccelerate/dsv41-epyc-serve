#!/usr/bin/env bash
set -euo pipefail
: "${EXL3_DIR:?Set EXL3_DIR to the DeepSeek V4.1 Flash 1017 EXL3 checkpoint directory}"
: "${FP8_DIR:?Set FP8_DIR to the matching original FP8 checkpoint directory}"
: "${ENGRAM_DIR:?Set ENGRAM_DIR to the matching Engram shard directory; it may equal FP8_DIR}"
mode=${DSV41_TRANSPORT:-p2p4}
case "$mode" in
  hostpair) host=1; pair=1; shm=0; label=hostpair ;;
  p2p4) host=0; pair=0; shm=1; label=nccl-p2p4 ;;
  *) echo 'DSV41_TRANSPORT must be hostpair or p2p4' >&2; exit 1 ;;
esac
payload=${DSV41_P2P_PAYLOAD_DTYPE:-native}
case "$payload" in native|fp32) ;; *) echo 'Payload dtype must be native or fp32' >&2; exit 1 ;; esac
image=${DSV41_IMAGE:-dsv41-optimized:20261030-transport-v3-$mode}
name=${DSV41_CONTAINER:-dsv41-optimized}
gpus=${DSV41_GPUS:-}
if [ -z "$gpus" ]; then
  mapfile -t cards < <(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F, '/CMP 170HX/ {gsub(/[[:space:]]/,"",$1); print $1}')
  if [ "${#cards[@]}" -ne 4 ]; then
    echo 'Expected four CMP 170HX cards; set DSV41_GPUS explicitly for another four-GPU SM80 selection.' >&2
    exit 1
  fi
  gpus="device=$(IFS=,; echo "${cards[*]}")"
fi
nodes=${DSV41_RANK_NODES:-0 0 0 0}
read -r -a rank_nodes <<< "$nodes"
if [ "${#rank_nodes[@]}" -ne 4 ]; then echo 'DSV41_RANK_NODES requires four node numbers' >&2; exit 1; fi
for directory in "$EXL3_DIR" "$FP8_DIR" "$ENGRAM_DIR"; do
  if [ ! -d "$directory" ]; then echo "Directory missing: $directory" >&2; exit 1; fi
done
if [ "$(docker image inspect --format '{{index .Config.Labels "io.dsv41.transport"}}' "$image")" != "$label" ] || [ "$(docker image inspect --format '{{index .Config.Labels "io.dsv41.startup-contract"}}' "$image")" != 'transport-v3' ]; then
  echo 'Image transport/revision does not match. Run install_p2p4.py first.' >&2
  exit 1
fi
exec docker run --name "$name" --init \
  --gpus "\"$gpus\"" --shm-size 2g --ulimit memlock=-1:-1 --cap-add SYS_NICE \
  --tmpfs /engram-resident:rw,size=216g,mode=0700 \
  --mount "type=bind,src=$(realpath "$EXL3_DIR"),dst=/exl3-1017,readonly" \
  --mount "type=bind,src=$(realpath "$FP8_DIR"),dst=/fp8-1017,readonly" \
  --mount "type=bind,src=$(realpath "$ENGRAM_DIR"),dst=/engram-1017,readonly" \
  -e "DSV41_TRANSPORT=$mode" -e "VLLM_HOST_ALLREDUCE=$host" -e "W15_PAIR_TRANSPORT=$pair" -e "NCCL_SHM_DISABLE=$shm" \
  -e "DSV41_P2P_PAYLOAD_DTYPE=$payload" -e VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS -e DSV41_REQUIRE_P2P=1 \
  -e "DSV41_RANK_NODES=$nodes" \
  -e "DSV41_ARENA_NODES=${rank_nodes[0]} ${rank_nodes[2]}" \
  -p "${DSV41_PORT:-8095}:8095" \
  "$image" serve "$@"
