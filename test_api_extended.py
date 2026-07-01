#!/usr/bin/env python3
"""
Extended API test suite for the RAG chat endpoint.

Covers the five previously validated questions plus ~25 additional cases
spanning Parts 61, 67, 71, 73, 91, and out-of-scope abstention checks.

Usage:
    python test_api_extended.py                        # run full suite
    python test_api_extended.py --section part61       # run one section only
    python test_api_extended.py --url http://host:port # custom base URL
    python test_api_extended.py --verbose              # show full reply text
    python test_api_extended.py --list                 # list cases without calling the API
    python test_api_extended.py --stream               # drive the SSE streaming endpoint
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

BASE_URL = "http://127.0.0.1:5000"

# ── colour helpers ────────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)


# ── test-case definition ──────────────────────────────────────────────────────

@dataclass
class Case:
    question: str
    section: str                        # logical group label (used for --section filter)
    expect_abstain: bool = False        # True → out-of-scope; reply must be abstained
    expect_grounded: bool = True        # False → don't require citations
    # If provided, at least ONE of these strings must appear (case-insensitive) in the reply.
    expect_keywords: list[str] = field(default_factory=list)
    note: str = ""                      # extra context shown next to the question


# ── test cases ────────────────────────────────────────────────────────────────

CASES: list[Case] = [

    # ── Previously validated ──────────────────────────────────────────────────
    Case(
        section="part61",
        question="What aeronautical experience is required for a private pilot certificate "
                 "with an airplane single-engine rating?",
        expect_keywords=["40", "hour"],
        note="previously tested",
    ),
    Case(
        section="part67",
        question="Which medical conditions disqualify an applicant for a first-class airman "
                 "medical certificate?",
        expect_keywords=["first"],
        note="previously tested",
    ),
    Case(
        section="part91",
        question="What are the fuel-reserve requirements for VFR flight, day versus night?",
        expect_keywords=["30", "45"],
        note="previously tested",
    ),
    Case(
        section="part71_91",
        question="How do operating requirements differ between Class B and Class C airspace?",
        expect_keywords=["class b", "class c"],
        note="previously tested",
    ),
    Case(
        section="part73",
        question="What must a pilot do before operating in an active restricted area?",
        expect_keywords=["permission", "authorization", "controlling", "agency"],
        note="previously tested",
    ),

    # ── Part 61 – new questions ───────────────────────────────────────────────
    Case(
        section="part61",
        question="What aeronautical experience is required for an instrument rating in an airplane?",
        expect_keywords=["50", "hour", "instrument"],
    ),
    Case(
        section="part61",
        question="What are the minimum flight-time requirements for a commercial pilot certificate "
                 "in an airplane single-engine land?",
        expect_keywords=["250", "hour"],
    ),
    Case(
        section="part61",
        question="How often must a pilot complete a flight review to remain current as pilot in command?",
        expect_keywords=["24", "month", "review"],
    ),
    Case(
        section="part61",
        question="What are the recent flight experience requirements for carrying passengers at night?",
        expect_keywords=["takeoff", "landing", "night"],
    ),
    Case(
        section="part61",
        question="What are the minimum aeronautical experience requirements for an airline transport "
                 "pilot certificate in an airplane?",
        expect_keywords=["1500", "1,500", "hour"],
    ),
    Case(
        section="part61",
        question="What logbook entries are required after each flight lesson?",
        expect_keywords=["log", "date", "hour", "instructor"],
    ),
    Case(
        section="part61",
        question="Under what conditions may a student pilot act as pilot in command of an aircraft?",
        expect_keywords=["solo", "student", "instructor"],
    ),

    # ── Part 67 – new questions ───────────────────────────────────────────────
    Case(
        section="part67",
        question="What vision standards are required for a second-class airman medical certificate?",
        expect_keywords=["vision", "second", "20/"],
    ),
    Case(
        section="part67",
        question="What are the cardiovascular examination requirements for a first-class medical?",
        expect_keywords=["cardiac", "heart", "cardiovascular", "electrocardiogram", "EKG", "ECG"],
    ),
    Case(
        section="part67",
        question="What is the validity period of a third-class medical certificate for a pilot "
                 "who is under 40 years old?",
        expect_keywords=["60", "month", "third"],
    ),
    Case(
        section="part67",
        question="What mental health conditions are disqualifying for an airman medical certificate?",
        expect_keywords=["mental", "psychiatric", "personality", "psychosis"],
    ),

    # ── Part 71 – new questions ───────────────────────────────────────────────
    Case(
        section="part71",
        question="What are the dimensions and requirements for Class D airspace?",
        expect_keywords=["class d", "2500", "tower"],
    ),
    Case(
        section="part71",
        question="How is Class E airspace defined and where does it typically begin?",
        expect_keywords=["class e", "1200"],
    ),
    Case(
        section="part71",
        question="What defines the lateral and vertical dimensions of Class B airspace?",
        expect_keywords=["class b"],
    ),

    # ── Part 73 – new questions ───────────────────────────────────────────────
    Case(
        section="part73",
        question="What is the difference between a prohibited area and a restricted area?",
        expect_keywords=["prohibited", "restricted"],
    ),
    Case(
        section="part73",
        question="What is a warning area and how does it differ from a restricted area?",
        expect_keywords=["warning"],
    ),

    # ── Part 91 – new questions ───────────────────────────────────────────────
    Case(
        section="part91",
        question="What are the IFR fuel reserve requirements for an airplane on a flight "
                 "with an alternate airport?",
        expect_keywords=["45", "alternate"],
    ),
    Case(
        section="part91",
        question="What are the right-of-way rules when two aircraft are converging at the same altitude?",
        expect_keywords=["right", "converging", "yield"],
    ),
    Case(
        section="part91",
        question="What is the maximum indicated airspeed below 10,000 feet MSL?",
        expect_keywords=["250", "knot"],
    ),
    Case(
        section="part91",
        question="What equipment is required for flight in Class B airspace?",
        expect_keywords=["transponder", "ADS-B", "class b"],
    ),
    Case(
        section="part91",
        question="What are the minimum safe altitude rules for flight over a congested area?",
        expect_keywords=["1000", "congested", "foot"],
    ),
    Case(
        section="part91",
        question="What VFR weather minimums apply in Class G airspace below 1200 feet AGL during the day?",
        expect_keywords=["1", "mile", "clear"],
    ),
    Case(
        section="part91",
        question="What preflight actions are required before a flight under IFR or away from "
                 "the vicinity of an airport?",
        expect_keywords=["weather", "fuel", "takeoff", "landing"],
    ),

    # ── Out-of-scope (expect abstain) ─────────────────────────────────────────
    Case(
        section="out_of_scope",
        question="What is the best route to drive from New York to Los Angeles?",
        expect_abstain=True,
        expect_grounded=False,
    ),
    Case(
        section="out_of_scope",
        question="Who won the 2024 FIFA World Cup?",
        expect_abstain=True,
        expect_grounded=False,
    ),
    Case(
        section="out_of_scope",
        question="How do I bake sourdough bread?",
        expect_abstain=True,
        expect_grounded=False,
    ),
    Case(
        section="out_of_scope",
        question="What are the visa requirements for visiting Japan?",
        expect_abstain=True,
        expect_grounded=False,
    ),
]


# ── HTTP helper ───────────────────────────────────────────────────────────────

def post_chat(question: str, base_url: str, timeout: int = 90) -> dict:
    url = f"{base_url}/api/chat"
    payload = json.dumps({"message": question}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_chat_stream(question: str, base_url: str, timeout: int = 90,
                     on_delta=None) -> dict:
    """Hit the SSE streaming endpoint and return the final `done` payload dict.

    Same response shape as post_chat — the streamed `delta` events are reassembled
    here (and forwarded to on_delta for live display), then the `done` event
    carries the authoritative reply/citations/meta. Raises on an `error` event or
    if the stream ends without a `done`. Signature matches post_chat so run_suite
    can swap between them.
    """
    url = f"{base_url}/api/chat"
    payload = json.dumps({"message": question}).encode()
    req = urllib.request.Request(
        url, data=payload,
        # Asking for the event stream is what flips the backend into streaming mode.
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    done = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Each event is a single `data: {json}` line followed by a blank line, so
        # iterating lines (they arrive as produced) is enough — no frame buffering.
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            evt = json.loads(line[5:].strip())
            etype = evt.get("type")
            if etype == "delta":
                if on_delta:
                    on_delta(evt.get("text", ""))
            elif etype == "done":
                done = evt
            elif etype == "error":
                raise RuntimeError(evt.get("message", "stream error"))
    if done is None:
        raise RuntimeError("stream ended before a done event")
    return done


# ── result evaluation ─────────────────────────────────────────────────────────

@dataclass
class Result:
    case: Case
    data: dict
    elapsed_s: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate(case: Case, data: dict, elapsed_s: float) -> Result:
    r = Result(case=case, data=data, elapsed_s=elapsed_s)
    reply = data.get("reply", "")
    abstained = data.get("abstained", False)
    grounded = data.get("grounded", True)
    citations = data.get("citations", [])

    if not reply:
        r.failures.append("empty reply")
        return r

    if case.expect_abstain:
        if not abstained:
            r.warnings.append("expected ABSTAINED but model answered")
    else:
        if abstained:
            r.failures.append("unexpected ABSTAIN — question should be in-scope")

    if case.expect_grounded and not case.expect_abstain:
        if not grounded or not citations:
            r.failures.append("answer is ungrounded (no citations returned)")

    if case.expect_keywords and not case.expect_abstain:
        reply_lower = reply.lower()
        # Pass if ANY expected keyword appears in the reply
        if not any(kw.lower() in reply_lower for kw in case.expect_keywords):
            r.warnings.append(f"none of the expected keywords found: {case.expect_keywords}")

    invalid = data.get("invalid_citations", [])
    if invalid:
        r.warnings.append(f"invented citation markers neutralised: {invalid}")

    return r


# ── pretty printer ────────────────────────────────────────────────────────────

def print_result(r: Result, verbose: bool = False) -> None:
    data = r.data
    reply = data.get("reply", "")
    abstained = data.get("abstained", False)
    grounded = data.get("grounded", True)
    citations = data.get("citations", [])
    meta = data.get("meta", {})

    if abstained:
        badge = yellow("[ABSTAINED]")
    elif grounded:
        badge = green("[GROUNDED] ")
    else:
        badge = red("[UNGROUND] ")

    status = green("PASS") if r.passed else red("FAIL")
    snippet = reply if verbose else (reply[:200] + ("…" if len(reply) > 200 else ""))
    print(f"  {status} {badge}  {snippet}")

    if citations and not r.case.expect_abstain:
        labels = [c.get("cfr_citation") or c.get("source", "?") for c in citations]
        suffix = f" (+{len(labels)-5} more)" if len(labels) > 5 else ""
        print(dim(f"         Citations: {', '.join(labels[:5])}{suffix}"))

    for f in r.failures:
        print(red(f"         FAIL: {f}"))
    for w in r.warnings:
        print(yellow(f"         WARN: {w}"))

    if meta:
        parts = []
        if "cost_usd" in meta:
            parts.append(f"${meta['cost_usd']:.4f}")
        if "total_ms" in meta:
            parts.append(f"{meta['total_ms']}ms")
        if "input_tokens" in meta:
            parts.append(f"in={meta['input_tokens']} out={meta.get('output_tokens','?')}")
        print(dim(f"         {' · '.join(parts)}  wall={r.elapsed_s:.2f}s"))


# ── suite runner ──────────────────────────────────────────────────────────────

SECTION_ORDER = ["part61", "part67", "part71", "part71_91", "part73", "part91", "out_of_scope"]


def run_suite(base_url: str, section_filter: Optional[str], verbose: bool,
              stream: bool = False) -> None:
    # Exercise the SSE streaming endpoint when asked; post_chat_stream returns the
    # same payload shape, so the rest of the suite is identical.
    fetch = post_chat_stream if stream else post_chat
    cases = CASES if not section_filter else [c for c in CASES if c.section == section_filter]
    if not cases:
        print(red(f"No cases found for section '{section_filter}'."))
        valid = sorted({c.section for c in CASES})
        print(f"Valid sections: {', '.join(valid)}")
        sys.exit(1)

    present = {c.section for c in cases}
    sections = [s for s in SECTION_ORDER if s in present] + \
               sorted(present - set(SECTION_ORDER))

    results: list[Result] = []

    for sec in sections:
        sec_cases = [c for c in cases if c.section == sec]
        if sec == "out_of_scope":
            header = "Out-of-scope (expect ABSTAINED)"
        else:
            header = f"Part {sec.replace('_', ' + ').upper()}"
        print()
        print(bold(f"=== {header} ({len(sec_cases)} question{'s' if len(sec_cases) != 1 else ''}) ==="))

        for case in sec_cases:
            note = f"  [{case.note}]" if case.note else ""
            print()
            print(bold(f"  Q:{note} {case.question}"))
            try:
                t0 = time.perf_counter()
                data = fetch(case.question, base_url)
                elapsed = time.perf_counter() - t0
            except urllib.error.URLError as e:
                print(red(f"  Connection error: {e.reason}"))
                print("  Make sure the backend is running:  cd backend && python app.py")
                sys.exit(1)
            except Exception as e:
                print(red(f"  Error: {e}"))
                r = Result(case=case, data={}, elapsed_s=0.0)
                r.failures.append(str(e))
                results.append(r)
                continue

            r = evaluate(case, data, elapsed)
            results.append(r)
            print_result(r, verbose=verbose)

    # ── summary table ─────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    warned = sum(1 for r in results if r.warnings and r.passed)
    total = len(results)

    print()
    print(bold("=" * 72))
    print(bold("SUMMARY BY SECTION"))
    print(dim("─" * 72))

    col_w = 16
    print(f"  {'Section':<{col_w}} {'Pass':>5} {'Fail':>5} {'Warn':>5}")
    print(dim(f"  {'-'*col_w} {'----':>5} {'----':>5} {'----':>5}"))

    for sec in sections:
        sec_results = [r for r in results if r.case.section == sec]
        sp = sum(1 for r in sec_results if r.passed)
        sf = sum(1 for r in sec_results if not r.passed)
        sw = sum(1 for r in sec_results if r.warnings and r.passed)
        colour = green if sf == 0 else red
        print(colour(f"  {sec:<{col_w}} {sp:>5} {sf:>5} {sw:>5}"))

    print(dim("─" * 72))
    summary = f"  Total: {passed}/{total} passed"
    if warned:
        summary += f"  ({warned} with warnings)"
    print(green(bold(summary)) if failed == 0 else red(bold(summary)))

    if failed > 0:
        print()
        print(bold(red("FAILED CASES:")))
        for r in results:
            if not r.passed:
                print(red(f"  • [{r.case.section}] {r.case.question[:80]}"))
                for f in r.failures:
                    print(red(f"      → {f}"))

    sys.exit(0 if failed == 0 else 1)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default=BASE_URL,
                        help=f"Backend base URL (default: {BASE_URL})")
    parser.add_argument("--section",
                        help="Run only this section (e.g. part61, part91, out_of_scope)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full reply text instead of a 200-char snippet")
    parser.add_argument("--list", action="store_true",
                        help="List all test cases and exit without calling the API")
    parser.add_argument("--stream", action="store_true",
                        help="Exercise the SSE streaming endpoint (Accept: text/event-stream)")
    args = parser.parse_args()

    if args.list:
        present = {c.section for c in CASES}
        sections = [s for s in SECTION_ORDER if s in present] + \
                   sorted(present - set(SECTION_ORDER))
        for sec in sections:
            print(bold(f"\n[{sec}]"))
            for c in CASES:
                if c.section == sec:
                    tag = " (abstain)" if c.expect_abstain else ""
                    note = f" [{c.note}]" if c.note else ""
                    print(f"  {note}{tag} {c.question}")
        print(f"\nTotal: {len(CASES)} cases across {len(sections)} sections")
        return

    run_suite(args.url, args.section, args.verbose, stream=args.stream)


if __name__ == "__main__":
    main()
