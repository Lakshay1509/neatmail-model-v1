#!/usr/bin/env python3
"""
Eval harness for the NeatMail email classifier.

Sends each labeled case in cases.json to the running /classify endpoint and
checks the predicted `category` (and optionally `response_required`) against the
expected label. Reports a per-case pass/fail table plus an accuracy summary, and
exits non-zero if the pass rate is below --threshold (so it can gate CI).

The goal is not just "did Pending Response fire on the right emails" but also
"did it NOT over-fire on the counter-cases" (newsletters/receipts/notifications
that merely mention payments or files). Both directions are scored.

Usage:
    # 1. start the API in another terminal (needs OPENAI/PINECONE/DASHBOARD keys):
    #    uvicorn main:app --host 0.0.0.0 --port 8000
    # 2. then:
    python eval/run_eval.py
    python eval/run_eval.py --base-url http://localhost:8000 --threshold 1.0
    python eval/run_eval.py --only swiggy-receipt-ask,forward-latest-file
    python eval/run_eval.py --json          # machine-readable output

Config via env or flags:
    NEATMAIL_BASE_URL   default http://localhost:8000
    DASHBOARD_API_KEY   the X-API-Key the server expects (required)
    NEATMAIL_EVAL_USER  user_id used for cases (default "eval-harness"); use one
                        with NO stored corrections so this measures the prompt,
                        not correction memory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "cases.json"

# ── tiny ANSI helpers (auto-disabled when not a tty) ──────────────────────
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def dim(s):    return _c("2", s)
def bold(s):   return _c("1", s)


def _norm(s: str) -> str:
    """Match the server's tag normalization so 'Pending Response' == 'pending-response'."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def load_cases(path: Path):
    data = json.loads(path.read_text())
    default_tags = data.get("default_tags", [])
    cases = data.get("cases", [])
    for case in cases:
        case["email"].setdefault("tags", default_tags)
    return cases


def classify(base_url: str, api_key: str, user_id: str, email: dict, timeout: float):
    payload = {
        "user_id": user_id,
        "subject": email["subject"],
        "from": email["from"],
        "bodySnippet": email["bodySnippet"],
        "tags": email["tags"],
        "sensitivity": email.get("sensitivity", "if actionable"),
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/classify",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def check(expect: dict, got: dict):
    """Return (passed: bool, reasons: list[str]) for one case."""
    reasons = []
    got_cat = got.get("category", "")

    if "category" in expect:
        allowed = expect["category"]
        allowed = [allowed] if isinstance(allowed, str) else allowed
        if not any(_norm(got_cat) == _norm(a) for a in allowed):
            reasons.append(f"category: expected {allowed}, got '{got_cat}'")

    if "not_category" in expect:
        forbidden = expect["not_category"]
        forbidden = [forbidden] if isinstance(forbidden, str) else forbidden
        if any(_norm(got_cat) == _norm(f) for f in forbidden):
            reasons.append(f"category: must NOT be '{got_cat}'")

    if "response_required" in expect:
        want = bool(expect["response_required"])
        have = bool(got.get("response_required", False))
        if want != have:
            reasons.append(f"response_required: expected {want}, got {have}")

    return (len(reasons) == 0, reasons)


def main():
    ap = argparse.ArgumentParser(description="Eval the NeatMail classifier against labeled cases.")
    ap.add_argument("--base-url", default=os.environ.get("NEATMAIL_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("DASHBOARD_API_KEY", ""))
    ap.add_argument("--user-id", default=os.environ.get("NEATMAIL_EVAL_USER", "eval-harness"))
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--only", default="", help="comma-separated case ids to run")
    ap.add_argument("--threshold", type=float, default=1.0, help="min pass rate to exit 0 (default 1.0)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    args = ap.parse_args()

    if not args.api_key:
        print(red("ERROR: no API key. Set DASHBOARD_API_KEY or pass --api-key."), file=sys.stderr)
        return 2

    cases = load_cases(Path(args.cases))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print(red("No cases to run."), file=sys.stderr)
        return 2

    results = []
    for case in cases:
        row = {"id": case["id"], "note": case.get("note", ""), "expect": case["expect"]}
        try:
            got = classify(args.base_url, args.api_key, args.user_id, case["email"], args.timeout)
            row["got"] = got
            row["passed"], row["reasons"] = check(case["expect"], got)
        except Exception as e:  # network/HTTP/JSON error -> the case errors, counts as fail
            row["got"] = None
            row["passed"] = False
            row["reasons"] = [f"request failed: {e}"]
        results.append(row)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    rate = passed / total if total else 0.0

    if args.json:
        print(json.dumps({"passed": passed, "total": total, "pass_rate": rate, "results": results}, indent=2))
        return 0 if rate >= args.threshold else 1

    # ── human report ──────────────────────────────────────────────────────
    print()
    print(bold(f"NeatMail classifier eval  ·  {args.base_url}  ·  user_id={args.user_id}"))
    print(dim("-" * 78))
    for r in results:
        tag = green("PASS") if r["passed"] else red("FAIL")
        got_cat = (r["got"] or {}).get("category", "-") if r["got"] else "-"
        print(f"  {tag}  {r['id']:<26} got={got_cat!r}")
        if not r["passed"]:
            for reason in r["reasons"]:
                print(f"        {yellow('↳ ' + reason)}")
            if r.get("note"):
                print(f"        {dim('note: ' + r['note'])}")
    print(dim("-" * 78))

    summary = f"{passed}/{total} passed  ({rate*100:.0f}%)"
    print(bold(green(summary)) if passed == total else bold(yellow(summary)))
    if rate < args.threshold:
        print(red(f"below threshold {args.threshold*100:.0f}% → exit 1"))
    print()

    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
