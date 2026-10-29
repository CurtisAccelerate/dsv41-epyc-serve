"""Refuse to package a tree that still carries host identity or secrets."""
from __future__ import annotations

import re
import sys
import json
from pathlib import Path

PATTERNS = (
    re.compile(r"192\.168\.\d+\.\d+"),
    re.compile(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"Users[/\\][^/\s\\]+"),
    re.compile(r"/mnt/"),
    re.compile(r"BEGIN [A-Z0-9 ]*PRIVATE KEY"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"),
    re.compile(r"unix://"),
)
SKIP_NAMES = {".git", "__pycache__", "share-check.py"}


def scan(root: Path) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in {".so", ".cubin", ".pyc"}:
            hits.append(f"{path}: binary artifacts are not part of this source image")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            hits.append(f"{path}: not utf-8 text")
            continue
        if path.name == "packages.json":
            # Distribution versions such as 10.2.0.50 are not host addresses.
            try:
                versions = json.loads(text)
            except ValueError:
                versions = None
            if (isinstance(versions, dict) and versions
                and all(isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", k)
                        and isinstance(v, str) and re.fullmatch(r"[0-9][A-Za-z0-9_.!+\-]*", v)
                        for k, v in versions.items())):
                text = json.dumps(list(versions))
        for pat in PATTERNS:
            if pat.search(text):
                hits.append(f"{path}: matched {pat.pattern}")
                break
    return hits


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    hits = scan(root)
    if hits:
        print("share check failed")
        for hit in hits:
            print(hit)
        return 1
    print(f"share check ok {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
