#!/usr/bin/env python3
"""
Backend API test script for the RAG chat endpoint.

Usage:
    python test_api.py                      # run all built-in test cases
    python test_api.py "your question here" # test a single question
    python test_api.py --url http://host:port "question"  # custom base URL
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:5000"

# ── colour helpers (skip if stdout is not a tty) ─────────────────────────────
_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def post_chat(question: str, base_url: str = BASE_URL, timeout: int = 90) -> dict:
    url = f"{base_url}/api/chat"
    payload = json.dumps({"message": question}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ── pretty printer ────────────────────────────────────────────────────────────

def print_result(question: str, data: dict, elapsed_s: float) -> bool:
    """Print the response and return True if the test looks healthy."""
    print()
    print(bold(f"Q: {question}"))
    print(dim("─" * 72))

    reply = data.get("reply", "")
    abstained = data.get("abstained", False)
    grounded = data.get("grounded", True)
    citations = data.get("citations", [])
    invalid = data.get("invalid_citations", [])
    meta = data.get("meta", {})

    # status badge
    if abstained:
        badge = yellow("[ABSTAINED]")
    elif grounded:
        badge = green("[GROUNDED]")
    else:
        badge = red("[UNGROUNDED]")

    print(f"{badge}  {reply[:300]}{'…' if len(reply) > 300 else ''}")

    # citations
    if citations:
        print()
        print(dim(f"  Citations ({len(citations)}):"))
        for c in citations:
            label = c.get("cfr_citation") or c.get("source", "?")
            print(dim(f"    [{c['n']}] {label}  (chunk {c.get('chunk_index', '?')})"))

    if invalid:
        print(red(f"  Invalid citation markers (neutralised): {invalid}"))

    # meta
    if meta:
        parts = []
        if "input_tokens" in meta:
            parts.append(f"in={meta['input_tokens']}")
        if "output_tokens" in meta:
            parts.append(f"out={meta['output_tokens']}")
        if "cost_usd" in meta:
            parts.append(f"${meta['cost_usd']:.4f}")
        if "total_ms" in meta:
            parts.append(f"{meta['total_ms']}ms total")
        if "retrieval_ms" in meta:
            parts.append(f"retrieval={meta['retrieval_ms']}ms")
        if "llm_ms" in meta:
            parts.append(f"llm={meta['llm_ms']}ms")
        print(dim(f"  Meta: {' · '.join(parts)}"))

    print(dim(f"  Wall time: {elapsed_s:.2f}s"))

    # health check: reply must be non-empty
    ok = bool(reply)
    if not ok:
        print(red("  FAIL: empty reply"))
    return ok


# ── built-in test cases ───────────────────────────────────────────────────────

IN_SCOPE = [
    "What does 14 CFR 91.3 say about the authority of the pilot in command?",
    "What are the fuel requirements for VFR flight during the day?",
    "When is a flight review required under part 61?",
]

OUT_OF_SCOPE = [
    "What's a good recipe for chocolate-chip cookies?",
    "Who won the 2024 FIFA World Cup?",
]


def run_suite(base_url: str) -> None:
    passed = 0
    failed = 0
    skipped = 0

    print(bold("\n=== In-scope questions (expect: GROUNDED or ABSTAINED) ==="))
    for q in IN_SCOPE:
        try:
            t0 = time.perf_counter()
            data = post_chat(q, base_url)
            elapsed = time.perf_counter() - t0
            ok = print_result(q, data, elapsed)
            if ok:
                passed += 1
            else:
                failed += 1
        except urllib.error.URLError as e:
            print(red(f"\nCould not connect to {base_url}: {e.reason}"))
            print("Make sure the backend is running:  cd backend && python app.py")
            sys.exit(1)
        except Exception as e:
            print(red(f"\nError: {e}"))
            failed += 1

    print(bold("\n=== Out-of-scope questions (expect: ABSTAINED) ==="))
    for q in OUT_OF_SCOPE:
        try:
            t0 = time.perf_counter()
            data = post_chat(q, base_url)
            elapsed = time.perf_counter() - t0
            ok = print_result(q, data, elapsed)
            abstained = data.get("abstained", False)
            if ok and abstained:
                passed += 1
            elif ok and not abstained:
                print(yellow("  WARN: expected ABSTAINED but got an answer"))
                skipped += 1
            else:
                failed += 1
        except urllib.error.URLError as e:
            print(red(f"\nCould not connect to {base_url}: {e.reason}"))
            sys.exit(1)
        except Exception as e:
            print(red(f"\nError: {e}"))
            failed += 1

    total = passed + failed + skipped
    print()
    print(bold("=" * 72))
    summary = f"Results: {passed}/{total} passed"
    if skipped:
        summary += f"  {skipped} warning(s)"
    print(green(summary) if failed == 0 else red(summary))
    sys.exit(0 if failed == 0 else 1)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="Single question to test")
    parser.add_argument("--url", default=BASE_URL, help=f"Backend base URL (default: {BASE_URL})")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON response")
    args = parser.parse_args()

    if args.question:
        try:
            t0 = time.perf_counter()
            data = post_chat(args.question, args.url)
            elapsed = time.perf_counter() - t0
        except urllib.error.URLError as e:
            print(red(f"Could not connect to {args.url}: {e.reason}"))
            print("Make sure the backend is running:  cd backend && python app.py")
            sys.exit(1)

        if args.raw:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print_result(args.question, data, elapsed)
    else:
        run_suite(args.url)


if __name__ == "__main__":
    main()
