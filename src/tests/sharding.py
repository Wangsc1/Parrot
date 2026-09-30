"""Opt-in CI sharding; local full-suite execution does not load this plugin."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.test_shards import partition


def pytest_addoption(parser):
    group = parser.getgroup("parrot-shards")
    group.addoption("--shard-index", type=int, required=True)
    group.addoption("--shard-count", type=int, required=True)
    group.addoption("--shard-report", required=True)


def pytest_configure(config):
    try:
        partition([], config.getoption("shard_index"), config.getoption("shard_count"))
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    complete = sorted(item.nodeid for item in items)
    if len(set(complete)) != len(complete):
        raise pytest.UsageError("duplicate nodeids cannot be sharded")
    index, count = config.getoption("shard_index"), config.getoption("shard_count")
    selected = partition(complete, index, count)
    selected_set = set(selected)
    deselected = [item for item in items if item.nodeid not in selected_set]
    items[:] = [item for item in items if item.nodeid in selected_set]
    config.hook.pytest_deselected(items=deselected)
    # Every xdist worker collects the same suite; xdist rejects disagreement.
    # A single writer avoids competing writes to the same report artifact.
    worker = getattr(config, "workerinput", {}).get("workerid")
    if worker in (None, "gw0"):
        path = Path(config.getoption("shard_report"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"index": index, "count": count,
                                    "collected": complete, "selected": selected}, indent=2) + "\n")
