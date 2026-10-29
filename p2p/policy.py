"""P2P policy for a GPU host.

The package default is all: the widest peer set, used to look for a speedup.
none is the measured serving stack and stays available as config/measured.env.

none
    Do not use GPU peer access. Host all-reduce, NCCL peer access off.
    This is the October 28 measured path, not the default of this package.

root-complex
    Peer only when NVIDIA topo says the two GPUs meet at PIX, PXB, or PHB.
    Those are the same PCIe switch or the same PCI host bridge. NODE and SYS
    stay off. On a single-socket EPYC with one GPU per root port, this mode
    enables nothing unless two GPUs actually share a host bridge.

all
    Peer for every pair whose topo link is PIX, PXB, PHB, NODE, SYS, or NVLink.
    This crosses separate root ports. The report still prints the link. A pair
    the driver later refuses is recorded as refused, not as enabled.
"""
from __future__ import annotations

CLOSE = ("PIX", "PXB", "PHB")
ANY = CLOSE + ("NODE", "SYS", "NVLINK")
MODES = ("none", "root-complex", "all")


def allowed(mode: str, link: str) -> bool:
    if mode not in MODES:
        raise ValueError(f"P2P mode must be one of {MODES}, got {mode!r}")
    link = link.upper()
    if mode == "none":
        return False
    if mode == "root-complex":
        return link in CLOSE
    return link in ANY


def nccl_env(mode: str) -> dict[str, str]:
    if mode == "none":
        return {"NCCL_P2P_DISABLE": "1"}
    if mode == "root-complex":
        return {"NCCL_P2P_DISABLE": "0", "NCCL_P2P_LEVEL": "PHB"}
    if mode == "all":
        return {"NCCL_P2P_DISABLE": "0", "NCCL_P2P_LEVEL": "SYS"}
    raise ValueError(mode)


def select(mode: str, links: dict[tuple[int, int], str]) -> list[dict]:
    rows = []
    for (a, b), link in sorted(links.items()):
        rows.append({
            "gpu_a": a,
            "gpu_b": b,
            "link": link.upper(),
            "enabled": allowed(mode, link),
        })
    return rows


def _undirected(pairs: list[tuple[int, int, str]]) -> dict[tuple[int, int], str]:
    out = {}
    for a, b, link in pairs:
        if a == b:
            continue
        out[(min(a, b), max(a, b))] = link.upper()
    return out


def selftest() -> None:
    # Four GPUs, each on its own root port, same socket. No shared host bridge.
    epyc = _undirected([
        (0, 1, "NODE"), (0, 2, "NODE"), (0, 3, "NODE"),
        (1, 2, "NODE"), (1, 3, "NODE"), (2, 3, "NODE"),
    ])
    # Two GPUs behind one host bridge, and one GPU on another root port.
    mixed = _undirected([
        (0, 1, "PHB"), (0, 2, "NODE"), (1, 2, "NODE"),
    ])
    assert [r["enabled"] for r in select("none", epyc)] == [False] * 6
    assert [r["enabled"] for r in select("root-complex", epyc)] == [False] * 6
    assert all(r["enabled"] for r in select("all", epyc))
    root = {(r["gpu_a"], r["gpu_b"]): r["enabled"] for r in select("root-complex", mixed)}
    assert root[(0, 1)] is True
    assert root[(0, 2)] is False and root[(1, 2)] is False
    assert all(r["enabled"] for r in select("all", mixed))
    nv = {(0, 1): "NVLINK"}
    assert select("all", nv)[0]["enabled"] is True
    assert select("root-complex", nv)[0]["enabled"] is False
    assert select("none", nv)[0]["enabled"] is False
    assert nccl_env("none")["NCCL_P2P_DISABLE"] == "1"
    assert nccl_env("root-complex")["NCCL_P2P_LEVEL"] == "PHB"
    assert nccl_env("all")["NCCL_P2P_LEVEL"] == "SYS"
    try:
        allowed("pairs", "PHB")
    except ValueError:
        pass
    else:
        raise SystemExit("unknown mode was accepted")


if __name__ == "__main__":
    selftest()
    print("p2p policy selftest ok")
