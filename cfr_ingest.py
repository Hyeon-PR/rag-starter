"""14 CFR ingestion — eCFR XML → structure-aware, citation-tagged chunks.

Fetches Title 14 (Aeronautics and Space) from the public eCFR API as clean
XML, one request per Part, parses the DIV5/DIV6/DIV8 hierarchy, and writes
structure-aware chunks to data/corpus.jsonl. Every chunk carries a canonical
`14 CFR § N` citation taken from the section's own number — so citations are
externally verifiable for free.

Why this instead of the PDFs in documents/:
  - The XML preserves Part -> Subpart -> Section structure and exact section
    numbers. The PDFs bury it behind page headers/footers and column layout.
  - A section (§) is the natural retrieval AND citation unit. We keep whole
    short sections together and split only long ones, so most chunks map 1:1
    to a citable §.

Dependency-light by design: standard library only (urllib + xml.etree + json),
so it runs in any Python 3.10+ without installing anything. The heavy deps
(embeddings) live in the *indexing* step, which consumes this file's output.

Usage:
    python cfr_ingest.py                  # all content Parts of Title 14
    python cfr_ingest.py --parts 1 73 91  # only these Parts (validation)
    python cfr_ingest.py --limit 5        # first 5 content Parts (quick smoke)
    python cfr_ingest.py --date 2026-06-08 --refresh
Outputs:
    data/ecfr_cache/part-<N>.xml   raw API responses (cached; safe to delete)
    data/corpus.jsonl              one JSON record per chunk (the corpus)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://www.ecfr.gov/api/versioner/v1"
TITLE = 14
UA = "rag-starter-cfr-ingest/1.0 (+https://github.com/Hyeon-PR/rag-starter)"

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "ecfr_cache"
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"

# Chunking knobs, in characters. A § is the unit: keep short sections whole,
# split long ones on paragraph boundaries with a little overlap so a fact that
# straddles a cut is still recoverable. ~2000 chars ≈ ~500 tokens.
TARGET_CHARS = 2000
OVERLAP_CHARS = 200

# Section sub-elements that are source/editorial noise, not regulatory text.
SKIP_TAGS = {"CITA", "SECAUTH", "EDNOTE", "EFFDNOT", "SOURCE", "AUTH", "FTNT", "EAR"}


# ── HTTP ────────────────────────────────────────────────────────────────────

def http_get(url: str, tries: int = 6) -> bytes:
    """GET with a polite User-Agent and exponential backoff on transient errors."""
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/xml, application/json"}
    )
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(min(30, 3 * (attempt + 1)))
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            if attempt < tries - 1:
                time.sleep(min(30, 3 * (attempt + 1)))
                continue
            raise
    raise last  # pragma: no cover — loop either returns or raises above


# ── Discovery ───────────────────────────────────────────────────────────────

def resolve_date(date: str | None) -> str:
    """Latest eCFR issue date for Title 14, unless an explicit date is given."""
    if date:
        return date
    data = json.loads(http_get(f"{API}/titles.json"))
    t = next(x for x in data["titles"] if x["number"] == TITLE)
    return t["latest_issue_date"]


def list_content_parts(date: str) -> list[str]:
    """All Part identifiers in Title 14 that carry content (skip Reserved/ranges)."""
    data = json.loads(http_get(f"{API}/structure/{date}/title-{TITLE}.json"))
    parts: list[str] = []
    seen: set[str] = set()

    def walk(node: dict) -> None:
        if node.get("type") == "part":
            ident = (node.get("identifier") or "").strip()
            desc = node.get("label_description") or node.get("label") or ""
            reserved = bool(node.get("reserved")) or "[reserved]" in desc.lower()
            # Keep plain integer Parts only ("61"); drop ranges ("50-59") and Reserved.
            if re.fullmatch(r"\d+", ident) and not reserved and ident not in seen:
                seen.add(ident)
                parts.append(ident)
        for child in node.get("children") or []:
            walk(child)

    walk(data)
    parts.sort(key=int)
    return parts


# ── Parsing ─────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return " ".join(s.split())


def _el_text(el: ET.Element) -> str:
    """Flatten an element to text like itertext(), but replace <img> figures with
    a visible placeholder so image-only regulatory content (diagrams, tables-as-
    images) leaves a traceable marker instead of vanishing silently."""
    if el.tag.lower() == "img":
        name = (el.get("src") or "").rsplit("/", 1)[-1]
        return f"[figure: {name}]" if name else "[figure]"
    out = [el.text or ""]
    for child in el:
        out.append(_el_text(child))
        out.append(child.tail or "")
    return "".join(out)


def _head(el: ET.Element) -> str:
    h = el.find("HEAD")
    return _norm(_el_text(h)) if h is not None else ""


def _section_citation(part: str, sec: str, heading: str) -> str:
    """Canonical citation for a section, honoring the Part's actual numbering.

    Standard Parts number sections with "§" and `sec` already encodes the Part
    (e.g. "61.3" -> "14 CFR § 61.3"). A few Parts use other styles where `sec`
    does NOT encode the Part — e.g. Part 241's "Section 03" / "Sec. 1-1", or
    "Table A to Part 117" — so we never fabricate a "§" for those.
    """
    h = heading.lstrip()
    if h.startswith("§"):
        return f"14 CFR § {sec}"
    m = re.match(r"(Sec\.|Section)\s+[\w.\-]+", h)
    if m:
        return f"14 CFR Part {part}, {m.group(0)}"
    m = re.match(r"(?:Table|Appendix)\s+\S+\s+to\s+Part\s+\d+", h)
    if m:
        return f"14 CFR {m.group(0)}"
    return f"14 CFR Part {part}, {sec}"


def _strip_amendments(div8: ET.Element) -> None:
    """Drop eCFR's 'Link to an amendment published at …' banners in place.

    These are <XREF> elements tagged with an amendment instruction (or <AMDDATE>
    nodes) — editorial metadata, not regulatory text. ElementTree has no parent
    pointers, so we walk each element and prune matching children.
    """
    for parent in div8.iter():
        for child in list(parent):
            is_amd_xref = child.tag == "XREF" and (
                "AMDINSN" in child.attrib or "AMDDATE" in child.attrib
            )
            if child.tag == "AMDDATE" or is_amd_xref:
                parent.remove(child)


def _section_body(div8: ET.Element) -> str:
    """Regulatory text of a section: its paragraphs in order, noise removed."""
    _strip_amendments(div8)
    out: list[str] = []
    for child in div8:
        if child.tag == "HEAD" or child.tag in SKIP_TAGS:
            continue
        txt = _norm(_el_text(child))
        if txt:
            out.append(txt)
    return "\n".join(out)


def iter_sections(part_xml: bytes):
    """Yield section AND appendix units from one Part's XML, with Part/Subpart context.

    Walks the single DIV5 in document order: each DIV6 (subpart) we pass updates
    the current subpart context; each DIV8 (section) is emitted with it; each DIV9
    (appendix — which also covers SFARs) is emitted at Part level (no subpart).
    Reserved units carry only a HEAD, yield an empty body, and are dropped.
    """
    root = ET.fromstring(part_xml)
    div5 = root if root.tag == "DIV5" else root.find(".//DIV5")
    if div5 is None:
        return
    part = (div5.get("N") or "").strip()
    part_head = _head(div5)
    subpart: str | None = None
    subpart_head: str | None = None
    # Materialize the traversal first: _section_body() prunes amendment nodes in
    # place, and mutating the tree under a live .iter() generator can skip
    # siblings. We only act on DIV6/DIV8/DIV9 (never removed), so this is safe.
    for el in list(div5.iter()):
        if el.tag == "DIV6" and el.get("TYPE") == "SUBPART":
            subpart = (el.get("N") or "").strip()
            subpart_head = _head(el)
        elif el.tag == "DIV8" and el.get("TYPE") == "SECTION":
            sec = (el.get("N") or "").strip()
            heading = _head(el) or f"§ {sec}"
            # Strip the leading reference ("§ 61.3" / "Section 03" / "Sec. 1-1")
            # to leave just the descriptive title.
            title = re.sub(r"^(?:§+|Sec\.|Section)\s*[\w.\-]+\.?\s*", "", heading).strip() or heading
            body = _section_body(el)
            if not body:
                continue
            yield {
                "kind": "section",
                "part": part,
                "part_heading": part_head,
                "subpart": subpart,
                "subpart_heading": subpart_head,
                "section": sec,
                "heading": heading,
                "title": title,
                "citation": _section_citation(part, sec, heading),
                "body": body,
            }
        elif el.tag == "DIV9" and el.get("TYPE") == "APPENDIX":
            # N is already a canonical label, e.g. "Appendix A to Part 91" or
            # "Special Federal Aviation Regulation No. 50-2". Appendices belong
            # to the Part, not a subpart.
            label = (el.get("N") or "").strip()
            heading = _head(el) or label
            title = heading.split("—", 1)[1].strip() if "—" in heading else heading
            body = _section_body(el)
            if not body:
                continue
            yield {
                "kind": "appendix",
                "part": part,
                "part_heading": part_head,
                "subpart": None,
                "subpart_heading": None,
                "section": label,
                "heading": heading,
                "title": title,
                "citation": f"14 CFR {label}",
                "body": body,
            }


# ── Chunking ────────────────────────────────────────────────────────────────

def _split_body(body: str, budget: int) -> list[str]:
    """Pack paragraphs into <= budget-char pieces with overlap; hard-split giants."""
    paras = [p for p in body.split("\n") if p.strip()]
    pieces: list[str] = []
    for p in paras:
        if len(p) <= budget:
            pieces.append(p)
        else:  # a single paragraph longer than the budget — window it
            for i in range(0, len(p), budget):
                pieces.append(p[i:i + budget])

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > budget:
            chunks.append(cur)
            # Seed the next chunk with overlap, but only as much as still fits
            # under budget — a hard-split piece can already fill the whole budget,
            # in which case there's no room for overlap.
            room = min(OVERLAP_CHARS, budget - len(piece) - 1)
            tail = cur[-room:] if room > 0 else ""
            cur = (tail + "\n" + piece) if tail else piece
        else:
            cur = (cur + "\n" + piece) if cur else piece
    if cur:
        chunks.append(cur)
    return chunks or [body]


def chunk_section(sec: dict) -> list[dict]:
    """One section -> one or more chunks, each prefixed with its citation+title.

    The citation/title prefix makes each chunk self-describing: the embedder
    sees which § it is, and the LLM sees the canonical cite inline.
    """
    ctx = f"{sec['citation']} — {sec['title']}".rstrip(" —")
    budget = max(400, TARGET_CHARS - len(ctx) - 1)
    pieces = _split_body(sec["body"], budget)
    n = len(pieces)
    records: list[dict] = []
    for i, piece in enumerate(pieces):
        text = f"{ctx}\n{piece}"
        records.append({
            "kind": sec["kind"],
            "cfr_citation": sec["citation"],
            "title": TITLE,
            "part": sec["part"],
            "part_heading": sec["part_heading"],
            "subpart": sec["subpart"],
            "subpart_heading": sec["subpart_heading"],
            "section": sec["section"],
            "heading": sec["heading"],
            "source": sec["citation"],   # frontend "Sources:" shows the canonical cite
            "chunk_index": i,
            "n_chunks_in_section": n,
            "text": text,
            "char_len": len(text),
        })
    return records


# ── Driver ──────────────────────────────────────────────────────────────────

def fetch_part(date: str, part: str, refresh: bool) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"part-{part}.xml"
    if cached.exists() and not refresh:
        return cached.read_bytes()
    data = http_get(f"{API}/full/{date}/title-{TITLE}.xml?part={part}")
    cached.write_bytes(data)
    return data


def build(parts: list[str], date: str, refresh: bool) -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CORPUS_PATH.with_name(CORPUS_PATH.name + ".tmp")
    chunk_id = 0
    total_units = 0
    skipped: list[str] = []   # fetch/parse failed — re-run to fill the gap
    empty: list[str] = []     # yielded no units (Reserved/edge layout)
    # Write to a temp file and os.replace() on success, so a crash mid-run can
    # never truncate a previously-good corpus.jsonl.
    with tmp.open("w", encoding="utf-8") as out:
        for idx, part in enumerate(parts, 1):
            was_cached = (CACHE_DIR / f"part-{part}.xml").exists() and not refresh
            try:
                xml = fetch_part(date, part, refresh)
            except Exception as e:  # one bad Part shouldn't kill the whole run
                print(f"  ! Part {part}: fetch failed ({e}) — skipped", file=sys.stderr)
                skipped.append(part)
                continue
            try:
                units = list(iter_sections(xml))
            except Exception as e:  # malformed XML / unexpected structure
                print(f"  ! Part {part}: parse failed ({e}) — skipped", file=sys.stderr)
                skipped.append(part)
                continue
            # Buffer the whole Part before writing, so a failure can't leave a
            # half-written Part in the output.
            lines: list[str] = []
            for sec in units:
                for rec in chunk_section(sec):
                    rec["chunk_id"] = chunk_id
                    lines.append(json.dumps(rec, ensure_ascii=False))
                    chunk_id += 1
            out.write("".join(line + "\n" for line in lines))
            total_units += len(units)
            if not units:
                empty.append(part)
            print(f"  [{idx}/{len(parts)}] Part {part}: {len(units)} units -> {len(lines)} chunks")
            if not was_cached:
                time.sleep(0.5)  # be polite only on real network hits
    os.replace(tmp, CORPUS_PATH)
    print(f"\n✓ {total_units} units -> {chunk_id} chunks  →  {CORPUS_PATH.relative_to(ROOT)}")
    if skipped:
        print(f"  ⚠ {len(skipped)} Part(s) FAILED — re-run to fill gaps: {', '.join(skipped)}")
    if empty:
        print(f"  · {len(empty)} Part(s) had 0 units (likely Reserved/edge): {', '.join(empty)}")
    return chunk_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest 14 CFR from eCFR into data/corpus.jsonl")
    ap.add_argument("--parts", nargs="+", help="Only these Part numbers (e.g. 1 73 91)")
    ap.add_argument("--limit", type=int, help="Only the first N content Parts")
    ap.add_argument("--date", help="eCFR issue date YYYY-MM-DD (default: latest)")
    ap.add_argument("--refresh", action="store_true", help="Re-fetch even if cached")
    args = ap.parse_args()

    date = resolve_date(args.date)
    print(f"14 CFR @ eCFR issue date {date}")
    if args.parts:
        parts = args.parts
    else:
        parts = list_content_parts(date)
        if args.limit:
            parts = parts[: args.limit]
    preview = ", ".join(parts[:20]) + (" …" if len(parts) > 20 else "")
    print(f"Ingesting {len(parts)} Part(s): {preview}\n")
    build(parts, date, args.refresh)


if __name__ == "__main__":
    main()
