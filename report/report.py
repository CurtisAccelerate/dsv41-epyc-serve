"""Print the host, the P2P decision, the graph plan, and the matched benches.

Read-only. This process does not set a power cap, enable peer access, or
open a model. Set DSV41_P2P to none, root-complex, or all before the run.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p2p.policy import MODES, nccl_env, select  # noqa: E402

# Package default: widest peer set. The measured stack is config/measured.env.
DEFAULT_P2P = "all"


def load_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=20)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def parse_topo(text: str) -> tuple[list[int], dict[tuple[int, int], str]]:
    text = ANSI.sub("", text)
    links: dict[tuple[int, int], str] = {}
    gpus: list[int] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("GPU"):
            continue
        try:
            src = int(parts[0][3:])
        except ValueError:
            continue
        if src not in gpus:
            gpus.append(src)
        col = 0
        for token in parts[1:]:
            if token in {"X", "CPU", "NUMA"} or token.startswith("CPU") or token.startswith("NUMA"):
                break
            if token.upper() in {"PIX", "PXB", "PHB", "NODE", "SYS", "NV1", "NV2", "NV3", "NV4", "NV5", "NV6", "NV7", "NV8", "NV12"}:
                if token.upper().startswith("NV"):
                    kind = "NVLINK"
                else:
                    kind = token.upper()
                a, b = sorted((src, col))
                if a != b:
                    links[(a, b)] = kind
                col += 1
            elif token == "X":
                col += 1
    return gpus, links


def gpu_rows() -> list[dict]:
    text = run(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,power.limit,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"])
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 8:
            continue
        rows.append({
            "index": int(cells[0]),
            "name": cells[1],
            "bus_id": cells[2],
            "power_limit_w": cells[3],
            "sm_clock_mhz": cells[4],
            "mem_clock_mhz": cells[5],
            "pcie_gen": cells[6],
            "pcie_width": cells[7],
        })
    return rows


def cpu_summary() -> dict:
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
    }


def section(title: str) -> None:
    print(f"\n== {title}")


def main() -> int:
    if "--selftest" in sys.argv:
        gpus, links = parse_topo(
            "        GPU0  GPU1  GPU2\n"
            "GPU0     X   PHB  NODE\n"
            "GPU1    PHB   X   NODE\n"
            "GPU2    NODE NODE   X\n"
        )
        assert gpus == [0, 1, 2]
        assert links[(0, 1)] == "PHB" and links[(0, 2)] == "NODE"
        colored = (
            "\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\x1b[0m\n"
            "GPU0\t X \tNODE\tNODE\tNODE\n"
            "GPU1\tNODE\t X \tNODE\tNODE\n"
            "GPU2\tNODE\tNODE\t X \tNODE\n"
            "GPU3\tNODE\tNODE\tNODE\t X \n"
        )
        _, epyc = parse_topo(colored)
        assert len(epyc) == 6 and set(epyc.values()) == {"NODE"}
        assert [r["enabled"] for r in select("root-complex", epyc)] == [False] * 6
        assert all(r["enabled"] for r in select("all", epyc))
        chosen = select("root-complex", links)
        assert {(r["gpu_a"], r["gpu_b"]): r["enabled"] for r in chosen}[(0, 1)] is True
        assert {(r["gpu_a"], r["gpu_b"]): r["enabled"] for r in chosen}[(0, 2)] is False
        measured = load_json("config/measured.json")
        assert DEFAULT_P2P == "all"
        assert measured["p2p"] == "none"
        assert measured["nccl"]["NCCL_P2P_DISABLE"] == "1"
        print("report selftest ok")
        return 0

    mode = os.environ.get("DSV41_P2P", DEFAULT_P2P)
    if mode not in MODES:
        print(f"DSV41_P2P must be one of {', '.join(MODES)}; got {mode}")
        return 2

    benches = load_json("benchmarks/matched.json")
    graphs = load_json("graphs/capture-plan.json")
    measured = load_json("config/measured.json")
    topo_text = run(["nvidia-smi", "topo", "-m"])
    gpus, links = parse_topo(topo_text or "")
    edges = select(mode, links) if links else []
    env = nccl_env(mode)

    report = {
        "package_default_p2p": DEFAULT_P2P,
        "p2p_mode": mode,
        "nccl_env": env,
        "measured": measured,
        "peer_access_changed": False,
        "power_cap_changed": False,
        "weights_included": False,
        "cpu": cpu_summary(),
        "gpus": gpu_rows(),
        "topo_links": [{"gpu_a": a, "gpu_b": b, "link": link} for (a, b), link in sorted(links.items())],
        "p2p_edges": edges,
        "graphs": graphs,
        "benchmarks": benches,
        "topo_raw_present": bool(topo_text),
    }

    section("identity")
    print("package default is p2p all: peer access across every card")
    print("measured stack is p2p none: host all-reduce, no GPU peers")
    print("weights, Engram tables, and the private runtime image are not in this image")
    print("this process prints the NCCL exports; it does not change a power cap")

    section("cpu")
    for key, value in report["cpu"].items():
        print(f"{key}: {value}")

    section("gpus")
    if not report["gpus"]:
        print("nvidia-smi did not return a GPU list")
    for gpu in report["gpus"]:
        print(
            f"GPU{gpu['index']} {gpu['name']} {gpu['bus_id']} "
            f"cap {gpu['power_limit_w']} SM {gpu['sm_clock_mhz']} mem {gpu['mem_clock_mhz']} "
            f"PCIe gen{gpu['pcie_gen']} x{gpu['pcie_width']}"
        )

    section(f"this run: p2p {mode}")
    if mode == "all":
        print("fastest theoretical peer set, and the package default")
    elif mode == "none":
        print("this run matches the measured stack")
    else:
        print("peers only inside one root complex")
    print("none: no GPU peer access. Measured serving stack. config/measured.env")
    print("root-complex: PIX, PXB, PHB only. NODE and SYS stay off. config/root-complex.env")
    print("all: PIX, PXB, PHB, NODE, SYS, and NVLINK. config/all.env")
    for key, value in env.items():
        print(f"export {key}={value}")
    if topo_text is None:
        print("nvidia-smi topo did not return a matrix; peer edges were not inferred")
    elif not edges:
        print("topo returned no GPU pairs; nothing to enable")
    for edge in edges:
        state = "enable" if edge["enabled"] else "leave off"
        print(f"GPU{edge['gpu_a']} -- GPU{edge['gpu_b']} {edge['link']}: {state}")

    section("measured settings")
    print(measured["note"])
    print(f"p2p {measured['p2p']} collective: {measured['collective']}")
    for key, value in measured["nccl"].items():
        print(f"measured export {key}={value}")
    print(
        f"tp {measured['tensor_parallel_size']} seqs {measured['max_num_seqs']} "
        f"batched tokens {measured['max_num_batched_tokens']} "
        f"model len {measured['max_model_len']} kv {measured['kv_cache_dtype']} "
        f"draft {measured['draft_depth']}"
    )
    print(measured["comparison_host"])
    for key, value in measured["kernel_by_users"].items():
        print(f"users {key}: {value}")

    section("optimization graphs")
    print(f"mode {graphs['cudagraph_mode']}")
    print("capture sizes " + " ".join(str(n) for n in graphs["capture_sizes"]))
    for row in graphs["plan"]:
        print(f"rows {row['rows']}: {row['expert_kernel']}; captured={row['captured']}")

    section("matched Morrow serving tok/s")
    print(benches["source"])
    for row in benches["morrow_serving_tok_s"]:
        print(f"users {row['users']} {row['path']}: count {row['count']} code {row['code']} prose {row['prose']}")

    section("matched Spark suite tok/s")
    print(benches["spark_suite_tok_s"]["note"])
    for row in benches["spark_suite_tok_s"]["october_28"]:
        mark = "serving" if row["serving_path"] else "not the serving path"
        print(f"users {row['users']}: aggregate {row['aggregate']} per-stream {row['per_stream']} ({mark})")

    section("json")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
