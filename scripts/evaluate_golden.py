#!/usr/bin/env python3
"""Evaluate ordered golden dataset turns against isolated local /chat instances.

Example: .venv/bin/python scripts/evaluate_golden.py --mode both --limit 2
Use --include-responses --output /private/path.json for manual rubric review.
Each mode creates a temporary SQLite database; no application data is touched.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personalized_memory.config import Settings
from personalized_memory.main import create_app

DEFAULT_DATASET = Path.home() / "Code" / "mynaksh_golden_dataset.json"


def assess_context(turn, result):
    """Only assess keys the API reports as used, not answer correctness."""
    used = set(result.get("context_used", []))
    missing = sorted(set(turn.get("expected_context_keys", [])) - used)
    unexpected = sorted(set(turn.get("excluded_context_keys", [])) & used)
    checks_present = bool(turn.get("expected_context_keys") or turn.get("excluded_context_keys"))
    return {
        "pass": (not missing and not unexpected) if checks_present else None,
        "missing": missing,
        "excluded_found": unexpected,
    }


def run_mode(mode, scenarios, settings_args, include_responses):
    records = []
    old_mode = os.environ.get("PMVA_MEMORY_FAST")
    os.environ["PMVA_MEMORY_FAST"] = "1" if mode == "fast" else "0"
    try:
        with tempfile.TemporaryDirectory(prefix="mynaksh-golden-") as temp:
            settings = Settings(database_path=Path(temp) / "golden.sqlite", **settings_args)
            with TestClient(create_app(settings=settings)) as client:
                for scenario in scenarios:
                    for index, turn in enumerate(scenario["turns"], 1):
                        payload = {key: turn[key] for key in ("user_id", "session_id", "message")}
                        start = time.perf_counter()
                        try:
                            response = client.post("/chat", json=payload)
                            elapsed_ms = round((time.perf_counter() - start) * 1000)
                            record = {
                                "scenario": scenario["name"],
                                "turn": index,
                                "status": response.status_code,
                                "latency_ms": elapsed_ms,
                                "expected_language": turn.get("expected_language"),
                                "expected_scope": turn.get("expected_scope"),
                                "subjective_review": "not evaluated",
                                "scope_review": "not observable from API response",
                            }
                            if response.status_code == 200:
                                body = response.json()
                                record["context"] = assess_context(turn, body)
                                record["context_used"] = body.get("context_used", [])
                                if include_responses:
                                    record.update(
                                        message=turn["message"],
                                        response=body["response"],
                                        rubric=turn.get("rubric"),
                                    )
                            records.append(record)
                            # Later turns depend on earlier state; stop this scenario on failure.
                            if response.status_code != 200:
                                break
                        except Exception as exc:
                            records.append(
                                {
                                    "scenario": scenario["name"],
                                    "turn": index,
                                    "error_type": type(exc).__name__,
                                    "latency_ms": round((time.perf_counter() - start) * 1000),
                                }
                            )
                            break
    finally:
        if old_mode is None:
            os.environ.pop("PMVA_MEMORY_FAST", None)
        else:
            os.environ["PMVA_MEMORY_FAST"] = old_mode
    latencies = [r["latency_ms"] for r in records if r.get("status") == 200]
    checks = [
        r["context"]["pass"] for r in records if "context" in r and r["context"]["pass"] is not None
    ]
    return {
        "mode": mode,
        "scenarios_requested": len(scenarios),
        "turns_attempted": len(records),
        "successful_turns": len(latencies),
        "context_checks_passed": sum(checks),
        "context_checks_total": len(checks),
        "latency_ms_median": round(statistics.median(latencies)) if latencies else None,
        "latency_ms_max": max(latencies, default=None),
        "turns": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("fast", "original", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios")
    parser.add_argument("--scenario", action="append", help="exact scenario name; repeatable")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--base-url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--output", type=Path, help="write full JSON report (mode 0600)")
    parser.add_argument(
        "--include-responses",
        action="store_true",
        help="include synthetic messages and replies in report",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.include_responses and not args.output:
        parser.error("--include-responses requires --output")
    scenarios = json.loads(args.dataset.read_text())
    if args.scenario:
        scenarios = [s for s in scenarios if s["name"] in args.scenario]
        missing = set(args.scenario) - {s["name"] for s in scenarios}
        if missing:
            parser.error(f"unknown scenarios: {', '.join(sorted(missing))}")
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        parser.error("no scenarios selected")
    # Match the voice gateway: the project's credential takes precedence over
    # unrelated credentials inherited from the shell.
    key = (
        dotenv_values(".env", interpolate=False).get("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        parser.error("Set LLM_API_KEY or OPENAI_API_KEY in the environment or .env")
    settings = {
        "llm_provider": "openai-compatible",
        "llm_model": args.model,
        "llm_base_url": args.base_url,
        "llm_api_key": key,
        "otel_exporter_otlp_endpoint": None,
    }
    modes = ("original", "fast") if args.mode == "both" else (args.mode,)
    results = [run_mode(mode, scenarios, settings, args.include_responses) for mode in modes]
    report = {
        "dataset": str(args.dataset),
        "model": args.model,
        "note": "Context checks cover API-reported keys only; language, scope, and rubric require review.",
        "results": results,
    }
    if args.output:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write("\n")
    summaries = [{k: v for k, v in result.items() if k != "turns"} for result in results]
    print(json.dumps(summaries, indent=2))
    expected = sum(len(s["turns"]) for s in scenarios)
    return int(any(r["successful_turns"] != expected for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
