"""The two CI shards together must run exactly the full collected suite."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.test_shards import partition, verify_reports


def _reports(tmp_path):
    complete = [f"test_fixture.py::test_value[{i:03}]" for i in range(100)]
    paths = []
    for index in (1, 2):
        path = tmp_path / f"shard-{index}.json"
        path.write_text(json.dumps({"index": index, "count": 2, "collected": complete,
                                    "selected": partition(complete, index, 2)}))
        paths.append(path)
    return paths


def test_partition_is_stable_exhaustive_and_disjoint(tmp_path):
    paths = _reports(tmp_path)
    result = verify_reports(paths, 2)
    assert result["collected"] == result["selected"] == 100
    complete = json.loads(paths[0].read_text())["collected"]
    assert set(partition(list(reversed(complete)), 1, 2)) == set(partition(complete, 1, 2))


@pytest.mark.parametrize("index,count", [(0, 2), (3, 2), (1, 0), (1, -1)])
def test_partition_rejects_invalid_shard(index, count):
    with pytest.raises(ValueError):
        partition([], index, count)


@pytest.mark.parametrize("fault", ["missing-report", "duplicate-report", "missing-test", "duplicate-test",
                                    "wrong-count", "different-collection", "empty-collection"])
def test_verification_rejects_incomplete_or_overlapping_shards(tmp_path, fault):
    paths = _reports(tmp_path)
    report = json.loads(paths[1].read_text())
    if fault == "missing-report":
        paths.pop()
    elif fault == "duplicate-report":
        paths[1] = paths[0]
    else:
        if fault == "missing-test":
            report["selected"].pop()
        elif fault == "duplicate-test":
            report["selected"].append(report["selected"][0])
        elif fault == "wrong-count":
            report["count"] = 3
        elif fault == "different-collection":
            report["collected"].pop()
        elif fault == "empty-collection":
            report["collected"] = []
        paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_reports(paths, 2)


def test_plugin_executes_each_case_once_with_xdist(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    suite = tmp_path / "test_shard_fixture.py"
    suite.write_text('import pytest\n@pytest.mark.parametrize("i", range(40))\n'
                     'def test_value(i):\n    assert i >= 0\n')
    reports = []
    for index in (1, 2):
        report = tmp_path / f"shard-{index}.json"
        result = subprocess.run([
            sys.executable, str(repo / "src/tests/isolated_pytest.py"), str(suite), "-q", "-n", "2",
            "-p", "src.tests.sharding", "--shard-index", str(index), "--shard-count", "2",
            "--shard-report", str(report),
        ], cwd=repo, env={**os.environ, "PYTEST_ADDOPTS": ""}, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        body = json.loads(report.read_text())
        assert f'{len(body["selected"])} passed' in result.stdout
        reports.append(report)
    assert verify_reports(reports, 2)["selected"] == 40
