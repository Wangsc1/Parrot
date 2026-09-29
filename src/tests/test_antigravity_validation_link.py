"""Offline tests for scripts/antigravity_validation_link.py (read-only helper).

All data is synthetic: a throwaway monthly log DB under tmp_path with the same
columns Parrot writes, and a VALIDATION_REQUIRED body shaped like the one
cloudcode-pa returns.  No network, no application config.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import sqlite3

from ._isolation import isolate

isolate()

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "antigravity_validation_link.py"
_mod = runpy.run_path(str(_SCRIPT))
extract_validation_url = _mod["extract_validation_url"]
with_authuser = _mod["with_authuser"]
email_from_channel_key = _mod["email_from_channel_key"]
collect = _mod["collect"]
monthly_dbs = _mod["monthly_dbs"]
main = _mod["main"]

EMAIL = "user.a+x@example.com"
OTHER = "user-b@example.com"
KEY = f"oauth:antigravity:{EMAIL}:project-a"
OTHER_KEY = f"oauth:antigravity:{OTHER}:project-b"
BASE = ("https://accounts.google.com/signin/continue?sarp=1&scc=1"
        "&continue=https://developers.google.com/gemini-code-assist/auth/auth_success_gemini"
        "&plt={plt}&flowName=GlifWebSignIn&authuser")


def _body(plt: str) -> str:
    url = BASE.format(plt=plt)
    payload = {"error": {
        "code": 403,
        "message": "Verify your account to continue.",
        "status": "PERMISSION_DENIED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "VALIDATION_REQUIRED",
             "domain": "cloudcode-pa.googleapis.com",
             "metadata": {"validation_url": url,
                          "validation_learn_more_url": "https://support.google.com/accounts?p=al_alert"}},
            {"@type": "type.googleapis.com/google.rpc.Help",
             "links": [{"description": "Verify your account", "url": url},
                       {"description": "Learn more", "url": "https://support.google.com/accounts?p=al_alert"}]},
        ],
    }}
    return "HTTP 403: " + json.dumps(payload, indent=2)


def _make_db(path: Path, rows_log, rows_retry) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE request_log (id INTEGER PRIMARY KEY, created_at REAL, final_channel_key TEXT,"
        " http_status INTEGER, error_message TEXT);"
        "CREATE TABLE retry_chain (id INTEGER PRIMARY KEY, channel_key TEXT, started_at REAL,"
        " outcome TEXT, error_detail TEXT);"
    )
    conn.executemany("INSERT INTO request_log (created_at, final_channel_key, http_status, error_message)"
                     " VALUES (?,?,?,?)", rows_log)
    conn.executemany("INSERT INTO retry_chain (channel_key, started_at, outcome, error_detail)"
                     " VALUES (?,?,?,?)", rows_retry)
    conn.commit()
    conn.close()


def _fixture(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir()
    _make_db(logs / "2026-08.db",
             [(100.0, KEY, 403, _body("OLD"))],
             [(KEY, 100.0, "http_auth_error", _body("OLD"))])
    _make_db(logs / "2026-09.db",
             [(200.0, KEY, 403, _body("NEW")), (150.0, OTHER_KEY, 403, _body("B1"))],
             [(KEY, 200.0, "http_auth_error", _body("NEW")),
              (OTHER_KEY, 150.0, "http_auth_error", _body("B1")),
              (OTHER_KEY, 180.0, "success", None),
              (KEY, 90.0, "success", None)])
    (logs / "notes.db").write_bytes(b"")  # non-monthly file must be ignored
    return logs


def test_extract_prefers_metadata_url_and_ignores_other_errors():
    assert extract_validation_url(_body("P1")) == BASE.format(plt="P1")
    assert extract_validation_url('HTTP 403: {"error": {"code": 403, "message": "forbidden"}}') is None
    assert extract_validation_url(None) is None
    # Truncated / non-JSON body still yields the sign-in link by regex.
    truncated = _body("P2")[:900]
    assert "VALIDATION_REQUIRED" in truncated
    assert extract_validation_url(truncated) == BASE.format(plt="P2")


def test_with_authuser_fills_bare_or_empty_param_only():
    bare = BASE.format(plt="P")
    filled = with_authuser(bare, EMAIL)
    assert filled.endswith("&authuser=user.a%2Bx%40example.com")
    assert filled.count("authuser") == 1
    assert with_authuser(bare + "=", EMAIL) == filled
    assert with_authuser(bare.replace("&authuser", "&authuser=0"), EMAIL) == filled
    # Everything before authuser is untouched.
    assert filled.startswith(bare[: -len("authuser")])
    assert with_authuser("https://accounts.google.com/signin/continue?plt=X", EMAIL).endswith(
        "?plt=X&authuser=user.a%2Bx%40example.com")
    assert with_authuser(bare, "") == bare


def test_email_from_channel_key():
    assert email_from_channel_key(KEY) == EMAIL
    assert email_from_channel_key(f"oauth:antigravity:{EMAIL}") == EMAIL
    assert email_from_channel_key(f"oauth:openai:{EMAIL}") == ""


def test_collect_newest_first_dedup_and_success_after(tmp_path):
    logs = _fixture(tmp_path)
    dbs = monthly_dbs(str(logs), 0)
    assert [Path(p).name for p in dbs] == ["2026-08.db", "2026-09.db"]
    assert [Path(p).name for p in monthly_dbs(str(logs), 1)] == ["2026-09.db"]

    hits, last_ok = collect(dbs)
    # request_log + retry_chain copies of one failure collapse to one hit.
    assert [(h.email, h.at) for h in hits] == [(EMAIL, 200.0), (OTHER, 150.0), (EMAIL, 100.0)]
    assert last_ok == {EMAIL: 90.0, OTHER: 180.0}

    only_a, _ = collect(dbs, EMAIL)
    assert {h.email for h in only_a} == {EMAIL}


def test_main_json_latest_per_account(tmp_path, capsys):
    logs = _fixture(tmp_path)
    assert main(["--log-dir", str(logs), "--months", "0", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["email"], r["succeededAfter"]) for r in rows] == [(EMAIL, False), (OTHER, True)]
    assert "plt=NEW" in rows[0]["url"] and rows[0]["url"].endswith("authuser=user.a%2Bx%40example.com")

    assert main(["--log-dir", str(logs), "--months", "0", "--json", "--all"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3


def test_main_text_and_exit_codes(tmp_path, capsys):
    logs = _fixture(tmp_path)
    assert main(["--log-dir", str(logs), "--email", EMAIL]) == 0
    out = capsys.readouterr().out
    assert "plt=NEW" in out and "plt=OLD" not in out
    assert main(["--log-dir", str(logs), "--email", "nobody@example.com"]) == 1
    assert main(["--log-dir", str(tmp_path / "missing")]) == 2


def test_script_is_read_only(tmp_path):
    logs = _fixture(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in logs.iterdir()}
    for db in logs.glob("2026-*.db"):
        db.chmod(0o444)
    logs.chmod(0o555)  # no journal/WAL files can be created either
    try:
        hits, _ = collect(monthly_dbs(str(logs), 0))
        assert hits
    finally:
        logs.chmod(0o755)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in logs.iterdir()}
    assert after == before
