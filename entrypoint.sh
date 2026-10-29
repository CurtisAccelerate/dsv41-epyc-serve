#!/bin/sh
# Precedence, highest first:
#   1. DSV41_P2P already set (docker -e or --env-file)
#   2. DSV41_P2P_FILE, a mounted env file, when DSV41_P2P is unset
#   3. config/all.env, the package default (peer access across every card)
set -eu
cd /opt/dsv41-serve
if [ -z "${DSV41_P2P:-}" ] && [ -n "${DSV41_P2P_FILE:-}" ] && [ -f "$DSV41_P2P_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$DSV41_P2P_FILE"
  set +a
fi
if [ -z "${DSV41_P2P:-}" ]; then
  set -a
  # shellcheck disable=SC1091
  . /opt/dsv41-serve/config/all.env
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
