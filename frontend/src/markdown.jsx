// Minimal, dependency-free markdown renderer for assistant answers, with
// built-in citation-token verification.
//
// Why hand-rolled: the frontend is intentionally React-only, and Claude's
// answers use a small, predictable slice of markdown (paragraphs, **bold**,
// *italic*, `code`, bullet/numbered lists, headings, blockquotes). We build
// React elements directly and NEVER use dangerouslySetInnerHTML, so answer
// text can never inject markup — the renderer is XSS-safe by construction.
//
// Not supported (rare in grounded CFR answers, kept out to stay small):
// tables, nested lists, images, raw HTML, and soft-wrapped list items (a bullet
// whose text continues on an unindented next line splits into separate blocks).
// Unsupported syntax degrades to plain text rather than breaking.

import { memo } from 'react'

const CITE_RE = /\[(\d+)\]/g

// Verify the [n] markers a final answer actually used against the citation
// set the backend returned. A marker is "verified" only if its number maps to
// a real source; anything else is a reference the model emitted that we must
// not present as a citation.
export function verifyCitations(text, citations) {
  const known = new Set((citations || []).map((c) => c.n))
  // Ignore markers inside inline code spans: the renderer shows those verbatim
  // (never as a citation), so counting them here would make the banner disagree
  // with what is actually rendered.
  const scanned = (text || '').replace(/`[^`\n]+`/g, '')
  const valid = new Set()
  const invalid = new Set()
  let total = 0
  for (const m of scanned.matchAll(CITE_RE)) {
    const n = Number(m[1])
    total += 1
    ;(known.has(n) ? valid : invalid).add(n)
  }
  // The backend rewrites any out-of-range/invented [n] to a literal [?] (see
  // app.py), so a raw numeric invalid rarely reaches us; count the [?] markers
  // as unresolved references so the banner still warns.
  const unresolved = (scanned.match(/\[\?\]/g) || []).length
  total += unresolved
  const asc = (a, b) => a - b
  return {
    total,
    valid: [...valid].sort(asc),
    invalid: [...invalid].sort(asc),
    unresolved,
  }
}

// One combined pass over inline syntax. Order matters: code first (so `*`
// inside code isn't treated as emphasis), then bold before italic (so `**`
// wins over `*`), then [n] citations.
const INLINE_RE =
  /(`[^`\n]+`)|(\*\*[^*\n]+\*\*|__[^_\n]+__)|(\*[^*\n]+\*|_[^_\n]+_)|\[(\d+)\]|(\[\?\])/g

function renderInline(text, ctx, keyPrefix) {
  const nodes = []
  let last = 0
  let k = 0
  // matchAll clones the regex internally, so nested renderInline() calls (for
  // emphasis, below) don't clobber a shared lastIndex — recursion is safe.
  for (const m of text.matchAll(INLINE_RE)) {
    if (m.index > last) nodes.push(text.slice(last, m.index))
    const key = `${keyPrefix}i${k++}`
    if (m[1]) {
      // Code stays verbatim (no emphasis/citation parsing inside it).
      nodes.push(<code key={key}>{m[1].slice(1, -1)}</code>)
    } else if (m[2]) {
      // Recurse so a [n] (or nested emphasis) inside **bold** is still rendered
      // as a real badge — otherwise the verification banner could flag a marker
      // the reader can't see.
      nodes.push(<strong key={key}>{renderInline(m[2].slice(2, -2), ctx, key)}</strong>)
    } else if (m[3]) {
      nodes.push(<em key={key}>{renderInline(m[3].slice(1, -1), ctx, key)}</em>)
    } else if (m[4] !== undefined) {
      const n = Number(m[4])
      const c = ctx.byNum.get(n)
      if (c) {
        nodes.push(
          <sup
            key={key}
            className="cite"
            aria-label={`Citation ${n}: ${c.source}, chunk ${c.chunk_index}`}
            title={`${c.source} · chunk ${c.chunk_index}`}
          >
            {n}
          </sup>,
        )
      } else {
        // Unverifiable reference: keep it visible but visibly flagged so it is
        // never mistaken for a real, checkable source.
        nodes.push(
          <sup
            key={key}
            className="cite invalid"
            aria-label={`Unverified reference ${n}: no matching source`}
            title="No matching source — unverified reference"
          >
            {n}?
          </sup>,
        )
      }
    } else {
      // Neutralized unresolved marker [?] emitted by the backend for an
      // out-of-range/invented reference — flag it, never show it as a source.
      nodes.push(
        <sup
          key={key}
          className="cite invalid"
          aria-label="Unverified reference: no matching source"
          title="No matching source — unverified reference"
        >
          ?
        </sup>,
      )
    }
    last = m.index + m[0].length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

// Soft line breaks inside one block become <br> (GitHub-flavored feel).
function renderSoft(text, ctx, keyPrefix) {
  const out = []
  text.split('\n').forEach((seg, si) => {
    if (si > 0) out.push(<br key={`${keyPrefix}br${si}`} />)
    out.push(...renderInline(seg, ctx, `${keyPrefix}l${si}`))
  })
  return out
}

const RE_HEADING = /^(#{1,6})\s+(.*)$/
const RE_BULLET = /^\s*[-*+]\s+(.*)$/
const RE_ORDERED = /^\s*\d+\.\s+(.*)$/
const RE_QUOTE = /^\s*>\s?(.*)$/
const RE_RULE = /^\s*([-*_])(?:\s*\1){2,}\s*$/ // ---, ***, ___

function parseBlocks(text) {
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  const blocks = []
  let i = 0
  const n = lines.length

  while (i < n) {
    const line = lines[i]
    if (line.trim() === '') {
      i++
      continue
    }
    if (RE_RULE.test(line)) {
      blocks.push({ type: 'hr' })
      i++
      continue
    }
    const h = RE_HEADING.exec(line)
    if (h) {
      blocks.push({ type: 'h', level: h[1].length, text: h[2] })
      i++
      continue
    }
    if (RE_BULLET.test(line)) {
      const items = []
      while (i < n && RE_BULLET.test(lines[i])) {
        items.push(RE_BULLET.exec(lines[i])[1])
        i++
      }
      blocks.push({ type: 'ul', items })
      continue
    }
    if (RE_ORDERED.test(line)) {
      const items = []
      while (i < n && RE_ORDERED.test(lines[i])) {
        items.push(RE_ORDERED.exec(lines[i])[1])
        i++
      }
      blocks.push({ type: 'ol', items })
      continue
    }
    if (RE_QUOTE.test(line)) {
      const q = []
      while (i < n && RE_QUOTE.test(lines[i])) {
        q.push(RE_QUOTE.exec(lines[i])[1])
        i++
      }
      blocks.push({ type: 'quote', text: q.join('\n') })
      continue
    }
    // Paragraph: consume until a blank line or the start of another block.
    const para = []
    while (
      i < n &&
      lines[i].trim() !== '' &&
      !RE_RULE.test(lines[i]) &&
      !RE_HEADING.test(lines[i]) &&
      !RE_BULLET.test(lines[i]) &&
      !RE_ORDERED.test(lines[i]) &&
      !RE_QUOTE.test(lines[i])
    ) {
      para.push(lines[i])
      i++
    }
    blocks.push({ type: 'p', text: para.join('\n') })
  }
  return blocks
}

// Memoized so a full App re-render (e.g. on every keystroke in the composer)
// doesn't re-parse the markdown of every prior answer. Effective only if the
// `citations` prop is a stable reference — App passes `m.citations` directly.
export const Markdown = memo(function Markdown({ text, citations }) {
  const ctx = { byNum: new Map((citations || []).map((c) => [c.n, c])) }
  const rendered = parseBlocks(text || '').map((b, bi) => {
    const key = `b${bi}`
    switch (b.type) {
      case 'hr':
        return <hr key={key} />
      case 'h': {
        // Demote by one so an answer heading sits just under the page's <h1>
        // (h1 -> h2 ...), keeping heading order unbroken for screen readers.
        const Tag = `h${Math.min(b.level + 1, 6)}`
        return <Tag key={key}>{renderInline(b.text, ctx, key)}</Tag>
      }
      case 'ul':
        return (
          <ul key={key}>
            {b.items.map((it, ii) => (
              <li key={ii}>{renderInline(it, ctx, `${key}-${ii}`)}</li>
            ))}
          </ul>
        )
      case 'ol':
        return (
          <ol key={key}>
            {b.items.map((it, ii) => (
              <li key={ii}>{renderInline(it, ctx, `${key}-${ii}`)}</li>
            ))}
          </ol>
        )
      case 'quote':
        return <blockquote key={key}>{renderSoft(b.text, ctx, key)}</blockquote>
      default:
        return <p key={key}>{renderSoft(b.text, ctx, key)}</p>
    }
  })
  return <>{rendered}</>
})
