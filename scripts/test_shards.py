#!/usr/bin/env python3
"""Deterministic pytest partition and fail-closed CI coverage verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def partition(nodeids, index, count):
    if count < 1 or not 1 <= index <= count:
        raise ValueError("shard index must be in 1..count")
    return [nodeid for nodeid in nodeids
            if int.from_bytes(hashlib.sha256(nodeid.encode()).digest()[:8], "big") % count == index - 1]


def verify_reports(paths, count):
    if count < 1 or len(paths) != count:
        raise ValueError("missing or extra shard reports")
    reports = [json.loads(Path(path).read_text()) for path in paths]
    indexes = [report["index"] for report in reports]
    if sorted(indexes) != list(range(1, count + 1)):
        raise ValueError("missing or duplicate shard index")
    complete = reports[0]["collected"]
    if not complete or len(set(complete)) != len(complete) or complete != sorted(complete):
        raise ValueError("invalid full collection")
    selected = []
    for report in reports:
        if report["count"] != count or report["collected"] != complete:
            raise ValueError("shards collected different test suites")
        expected = partition(complete, report["index"], count)
        if not expected or report["selected"] != expected:
            raise ValueError("shard does not match its deterministic partition")
        selected.extend(report["selected"])
    if sorted(selected) != complete:
        raise ValueError("shard union differs from full suite or contains duplicates")
    return {"collected": len(complete), "selected": len(selected),
            "shards": {str(report["index"]): len(report["selected"]) for report in reports}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    try:
        result = verify_reports(args.reports, args.count)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Shard coverage verification failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
