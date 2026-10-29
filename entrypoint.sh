#!/bin/sh
# config/all.env is the launch file: optimized path on, peer access on for four cards.
# A variable already set in the environment wins. DSV41_P2P_FILE overrides after that.
set -eu
cd /opt/dsv41-serve

load_if_unset() {
  file=$1
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key=${line%%=*}
    val=${line#*=}
    if ! eval "[ \"\${$key+x}\" = x ]"; then
      export "$key=$val"
    fi
  done < "$file"
}

load_if_unset /opt/dsv41-serve/config/all.env
if [ -n "${DSV41_P2P_FILE:-}" ] && [ -f "$DSV41_P2P_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$DSV41_P2P_FILE"
  set +a
fi

cmd=${1:-report}
case "$cmd" in
  report)
    exec python3 /opt/dsv41-serve/report/report.py
    ;;
  check)
    python3 /opt/dsv41-serve/share-check.py /opt/dsv41-serve
    python3 /opt/dsv41-serve/p2p/policy.py
    python3 /opt/dsv41-serve/report/report.py --selftest
    ;;
  *)
    echo "usage: report | check" >&2
    exit 2
    ;;
esac
