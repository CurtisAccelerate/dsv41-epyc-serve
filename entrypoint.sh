#!/bin/sh
# The image already exports config/all.env. Fill any hole from that file,
# then let DSV41_P2P_FILE replace values. An explicit DSV41_P2P stays put
# unless the file sets it again.
set -eu
cd /opt/dsv41-serve
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  key=${line%%=*}
  eval "isset=\${$key+x}"
  if [ -z "$isset" ]; then
    export "$line"
  fi
done < /opt/dsv41-serve/config/all.env
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
