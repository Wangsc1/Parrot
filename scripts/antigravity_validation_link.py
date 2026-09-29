#!/usr/bin/env python3
"""Print the latest Google validation link for Antigravity OAuth accounts.

When Google asks an Antigravity account to verify itself, cloudcode-pa returns
HTTP 403 ``VALIDATION_REQUIRED`` with a one-time ``validation_url`` in the
error details.  Parrot already stores that response in the monthly request
log; this script only reads it back and fills in ``authuser`` so the link opens
for the right Google account.

SQLite databases are opened with ``mode=ro``: no business records are written,
no network, no application imports, no config access. SQLite may create/update
WAL/SHM sidecars while reading a WAL database. Standard library only, so it also runs
from stdin inside the container::

    docker exec -i parrot python3 - < scripts/antigravity_validation_link.py
    docker exec -i parrot python3 - --email user@gmail.com < scripts/antigravity_validation_link.py
    python3 scripts/antigravity_validation_link.py --data-dir /opt/parrot/data

The printed link is a one-time sign-in continuation for that account; do not
paste it into public places.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
import sqlite3
import sys
from typing import Any, Iterable
from urllib.parse import quote, unquote_plus, urlsplit

CHANNEL_PREFIX = "oauth:antigravity:"
_MONTH_DB = re.compile(r"^\d{4}-\d{2}\.db$")
_SIGNIN_URL = re.compile(r"https://accounts\.google\.com/signin/continue[^\s\"'\\<>]+")
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass(frozen=True)
class ValidationHit:
    email: str
    channel_key: str
    at: float
    url: str
    source: str


def email_from_channel_key(channel_key: str) -> str:
    """``oauth:antigravity:<email>:<project_id>`` -> ``<email>``."""
    rest = str(channel_key or "")[len(CHANNEL_PREFIX):] if str(channel_key or "").startswith(CHANNEL_PREFIX) else ""
    return rest.split(":", 1)[0]


def _decode_error_json(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return value


def _is_validation_url(value: Any) -> bool:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.netloc == "accounts.google.com"
            and parsed.path == "/signin/continue" and bool(parsed.query))


def extract_validation_url(error_text: str | None) -> str | None:
    """Extract a complete URL, never guess the missing tail of a truncated value."""
    text = str(error_text or "")
    if "VALIDATION_REQUIRED" not in text.upper():
        return None
    payload = _decode_error_json(text)
    error = payload.get("error") if isinstance(payload, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        metadata = detail.get("metadata")
        candidate = metadata.get("validation_url") if isinstance(metadata, dict) else None
        if _is_validation_url(candidate):
            return candidate
        links = detail.get("links")
        for link in links if isinstance(links, list) else []:
            candidate = link.get("url") if isinstance(link, dict) else None
            if _is_validation_url(candidate):
                return candidate
    # A log may truncate the JSON after a complete URL string. Decode only
    # closed strings, including JSON escapes; an unfinished URL is not usable.
    for match in _JSON_STRING.finditer(text):
        try:
            candidate = json.loads(match.group(0))
        except ValueError:
            continue
        if _is_validation_url(candidate):
            return candidate
    # Non-JSON text needs an observed terminator too. EOF alone cannot tell a
    # complete URL from a token cut in half by the log's character limit.
    for match in _SIGNIN_URL.finditer(text):
        before = text[match.start() - 1:match.start()] if match.start() else ""
        after = text[match.end():match.end() + 1]
        bounded = (after == before if before in {"'", '"'} else bool(after and after.isspace()))
        if bounded and _is_validation_url(match.group(0)):
            return match.group(0)
    return None


def with_authuser(url: str, email: str) -> str:
    """Set ``authuser=<email>`` and leave the rest of the URL byte-identical."""
    if not email:
        return url
    value = "authuser=" + quote(email, safe="")
    before_fragment, fragment_sep, fragment = url.partition("#")
    base, _, query = before_fragment.partition("?")
    fields = query.split("&") if query else []
    updated = []
    found = False
    for field in fields:
        if unquote_plus(field.partition("=")[0]) == "authuser":
            if not found:
                updated.append(value)
                found = True
        else:
            updated.append(field)
    if not found:
        updated.append(value)
    # Split raw delimiters instead of re-encoding query values: signed tokens,
    # nested continue URLs and fragments must remain byte-identical.
    return base + "?" + "&".join(updated) + fragment_sep + fragment


def _connect_ro(path: str) -> sqlite3.Connection:
    uri = "file:" + quote(os.path.abspath(path)) + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5)


def _rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
    try:
        return conn.execute(sql, tuple(params)).fetchall()
    except sqlite3.OperationalError as exc:
        # A legacy schema may lack a source. Corruption, locks and I/O failures
        # are not an empty result and must reach the CLI's explicit error path.
        if str(exc).lower().startswith(("no such table:", "no such column:")):
            return []
        raise


def monthly_dbs(log_dir: str, months: int) -> list[str]:
    if months < 0:
        raise ValueError("--months 必须为非负整数（0 = 全部）")
    names = sorted(n for n in os.listdir(log_dir) if _MONTH_DB.match(n)) if os.path.isdir(log_dir) else []
    picked = names[-months:] if months > 0 else names
    return [os.path.join(log_dir, n) for n in picked]


def collect(db_paths: list[str], email: str | None = None, *,
            unusable_accounts: set[str] | None = None) -> tuple[list[ValidationHit], dict[str, float]]:
    """Return VALIDATION_REQUIRED hits (newest first) and last success per email."""
    prefix = CHANNEL_PREFIX + (email + ":" if email else "")
    like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    hits: dict[tuple[str, str], ValidationHit] = {}
    last_ok: dict[str, float] = {}
    for path in db_paths:
        with closing(_connect_ro(path)) as conn:
            sources = (
                ("retry_chain", "SELECT channel_key, started_at, error_detail FROM retry_chain "
                 "WHERE channel_key LIKE ? ESCAPE '\\' AND error_detail LIKE '%VALIDATION_REQUIRED%'"),
                ("request_log", "SELECT final_channel_key, created_at, error_message FROM request_log "
                 "WHERE final_channel_key LIKE ? ESCAPE '\\' AND error_message LIKE '%VALIDATION_REQUIRED%'"),
            )
            for source, sql in sources:
                for key, at, text in _rows(conn, sql, (like,)):
                    url = extract_validation_url(text)
                    addr = email_from_channel_key(key)
                    if not url:
                        if addr and unusable_accounts is not None:
                            unusable_accounts.add(addr)
                        continue
                    if not addr or at is None:
                        continue
                    hit = ValidationHit(addr, key, float(at), url, source)
                    # A URL can appear in both tables and recur later. Keep its
                    # newest occurrence, independent of DB/source/row order.
                    previous = hits.get((addr, url))
                    if previous is None or hit.at > previous.at:
                        hits[(addr, url)] = hit
            ok_sql = ("SELECT channel_key, MAX(started_at) FROM retry_chain "
                      "WHERE channel_key LIKE ? ESCAPE '\\' AND outcome = 'success' GROUP BY channel_key")
            for key, at in _rows(conn, ok_sql, (like,)):
                addr = email_from_channel_key(key)
                if addr and at is not None:
                    last_ok[addr] = max(last_ok.get(addr, 0.0), float(at))
    ordered = sorted(hits.values(), key=lambda h: h.at, reverse=True)
    return ordered, last_ok


def latest_per_account(hits: list[ValidationHit]) -> list[ValidationHit]:
    seen: set[str] = set()
    out = []
    for hit in hits:
        if hit.email not in seen:
            seen.add(hit.email)
            out.append(hit)
    return out


def default_data_dir() -> str:
    env = os.environ.get("ANTHROPIC_PROXY_DATA_DIR")
    if env:
        return env
    if os.path.isdir("/app/data/logs"):
        return "/app/data"
    # Match src.config.DATA_DIR for a source checkout, without importing it or
    # reading config.json. Standalone copies/stdin resolve relative to the cwd.
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(__file__) not in {"-", "<stdin>"} and os.path.isfile(os.path.join(source_root, "src", "config.py")):
        return source_root
    return os.getcwd()


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _ago(ts: float, now: float) -> str:
    minutes = max(0, int((now - ts) // 60))
    if minutes < 60:
        return f"{minutes} 分钟前"
    if minutes < 48 * 60:
        return f"{minutes // 60} 小时前"
    return f"{minutes // 1440} 天前"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读：从 Parrot 请求日志取 Antigravity 账号的 Google 验证链接")
    parser.add_argument("--data-dir", default=None, help="Parrot 数据目录（默认 $ANTHROPIC_PROXY_DATA_DIR、/app/data、源码根目录；独立脚本/stdin 用当前目录）")
    parser.add_argument("--log-dir", default=None, help="日志目录（默认 <data-dir>/logs；自定义了 logDir 时指定）")
    parser.add_argument("--email", default=None, help="只看这个账号")
    parser.add_argument("--months", type=int, default=2, help="扫描最近几个月的日志库（默认 2，0 = 全部）")
    parser.add_argument("--all", action="store_true", help="列出全部记录，而不是每个账号只给最新一条")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args(argv)
    if args.months < 0:
        parser.error("--months 必须为非负整数（0 = 全部）")

    log_dir = args.log_dir or os.path.join(args.data_dir or default_data_dir(), "logs")
    email = (args.email or "").strip() or None
    unusable_accounts: set[str] = set()
    try:
        dbs = monthly_dbs(log_dir, args.months)
        if not dbs:
            print(f"没有找到日志库：{log_dir}/YYYY-MM.db（用 --data-dir 或 --log-dir 指定）", file=sys.stderr)
            return 2
        hits, last_ok = collect(dbs, email, unusable_accounts=unusable_accounts)
    except (OSError, sqlite3.Error) as exc:
        print(f"读取日志失败（{log_dir}）：{exc}。请检查日志库及访问权限；这不表示没有验证记录。", file=sys.stderr)
        return 2
    if unusable_accounts:
        print("警告：以下账号的部分验证记录缺少完整有效链接（可能已截断）："
              + ", ".join(sorted(unusable_accounts))
              + "。仅展示可完整提取的链接，不保证对应最新一次验证要求；请核对记录时间。", file=sys.stderr)
    shown = hits if args.all else latest_per_account(hits)
    now = datetime.now(timezone.utc).timestamp()

    rows = []
    for hit in shown:
        ok_at = last_ok.get(hit.email)
        rows.append({
            "email": hit.email,
            "at": datetime.fromtimestamp(hit.at, tz=timezone.utc).isoformat(),
            "succeededAfter": bool(ok_at and ok_at > hit.at),
            "url": with_authuser(hit.url, hit.email),
        })

    if args.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0 if rows else 1
    if not rows:
        if unusable_accounts:
            print("日志中有 VALIDATION_REQUIRED 记录，但没有可完整提取的验证链接；请获取完整错误记录。")
        else:
            print("最近的日志里没有 VALIDATION_REQUIRED 记录。" + (f"（账号 {email}）" if email else ""))
            print("账号刚被要求验证但还没有请求落到它上面时，日志里不会有链接；给它发 1 次请求后再运行。")
        return 1
    for row, hit in zip(rows, shown):
        print(f"账号: {row['email']}")
        print(f"时间: {_fmt_time(hit.at)}（{_ago(hit.at, now)}）")
        if row["succeededAfter"]:
            print("状态: 此后该账号已有成功请求，这条链接很可能已用过或不再需要")
        print(f"链接: {row['url']}")
        print()
    print("提示：链接一次性有效，只登录目标账号的无痕窗口中打开；不要发到公开场合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
