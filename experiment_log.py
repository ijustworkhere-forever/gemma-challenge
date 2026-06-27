#!/usr/bin/env python3
"""
SQLite experiment ledger for local Gemma Challenge runs.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("data/experiments.sqlite3")


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    submission_path TEXT,
    run_uri TEXT,
    job_id TEXT,
    tps REAL,
    ppl REAL,
    latency REAL,
    status TEXT NOT NULL DEFAULT 'planned',
    notes TEXT,
    summary_json TEXT
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def load_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_metric(summary: dict[str, Any] | None, *names: str) -> Any:
    if not summary:
        return None
    for name in names:
        if name in summary:
            return summary[name]
    result = summary.get("result")
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
    return None


def add(args: argparse.Namespace) -> int:
    summary = load_summary(args.summary)
    tps = args.tps if args.tps is not None else pick_metric(summary, "tps", "tokens_per_second")
    ppl = args.ppl if args.ppl is not None else pick_metric(summary, "ppl", "perplexity")
    latency = args.latency if args.latency is not None else pick_metric(summary, "latency", "latency_mean")

    with connect(Path(args.db)) as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                name, target, submission_path, run_uri, job_id, tps, ppl,
                latency, status, notes, summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                args.name,
                args.target,
                args.submission,
                args.run_uri,
                args.job_id,
                tps,
                ppl,
                latency,
                args.status,
                args.notes,
                json.dumps(summary, sort_keys=True) if summary else None,
            ),
        )
        conn.commit()

    print(f"Logged experiment: {args.name}")
    return 0


def list_runs(args: argparse.Namespace) -> int:
    with connect(Path(args.db)) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, name, target, status, tps, ppl, job_id
            FROM experiments
            ORDER BY id DESC
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()

    if not rows:
        print("No experiments logged.")
        return 0

    for row in rows:
        id_, created_at, name, target, status, tps, ppl, job_id = row
        tps_text = f"{tps:.2f}" if tps is not None else "-"
        ppl_text = f"{ppl:.4f}" if ppl is not None else "-"
        job_text = job_id or "-"
        print(f"{id_:>4} {created_at} {status:<10} {target:<18} TPS={tps_text:<8} PPL={ppl_text:<8} {job_text} {name}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Track Gemma Challenge experiments in SQLite.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    subparsers = parser.add_subparsers(required=True)

    add_parser = subparsers.add_parser("add", help="Log an experiment")
    add_parser.add_argument("--name", required=True)
    add_parser.add_argument("--target", required=True, choices=["local_cpu", "local_3050ti", "zerogpu", "hf_a10g"])
    add_parser.add_argument("--submission", default="submissions/vllm_baseline")
    add_parser.add_argument("--run-uri")
    add_parser.add_argument("--job-id")
    add_parser.add_argument("--summary", help="Path to official summary.json")
    add_parser.add_argument("--tps", type=float)
    add_parser.add_argument("--ppl", type=float)
    add_parser.add_argument("--latency", type=float)
    add_parser.add_argument("--status", default="planned")
    add_parser.add_argument("--notes")
    add_parser.set_defaults(func=add)

    list_parser = subparsers.add_parser("list", help="List recent experiments")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.set_defaults(func=list_runs)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
