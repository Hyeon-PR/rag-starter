import { useCallback, useEffect, useRef, useState } from 'react'

import { Markdown, verifyCitations } from './markdown.jsx'

// Monotonic message ids (stable React keys without depending on array index).
let idSeq = 0
const nextId = () => ++idSeq

const EXAMPLES = [
  'What does 14 CFR 91.3 say about the authority of the pilot in command?',
  'What are the requirements to be issued a private pilot certificate?',
  'When is a flight review required under part 61?',
  'What are the fuel requirements for VFR flight during the day?',
  "What's a good recipe for chocolate-chip cookies?",
]

// Anthropic + retrieval can take a while; give it room but don't hang forever.
const REQUEST_TIMEOUT_MS = 90_000

function fmtMs(ms) {
  if (ms == null) return null
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`
}

// Per-answer cost + latency, from the backend's `meta`. Tokens are absent on the
// abstain path (no LLM call), so guard on each field.
function Metrics({ meta }) {
  const head = []
  if (meta.input_tokens != null) head.push(`${meta.input_tokens} in`)
  if (meta.output_tokens != null) head.push(`${meta.output_tokens} out`)
  if (meta.cost_usd != null) head.push(`$${meta.cost_usd.toFixed(4)}`)
  if (meta.total_ms != null) head.push(`${fmtMs(meta.total_ms)} total`)
  if (head.length === 0) return null

  const detail = []
  if (meta.retrieval_ms != null) detail.push(`retrieval ${fmtMs(meta.retrieval_ms)}`)
  if (meta.llm_ms != null) detail.push(`llm ${fmtMs(meta.llm_ms)}`)

  return (
    <div className="metrics" title={meta.model ? `model: ${meta.model}` : undefined}>
      {head.join(' · ')}
      {detail.length > 0 && <span className="metrics-detail"> ({detail.join(' · ')})</span>}
    </div>
  )
}

// Sources under an answer. Each cited chunk shows its CFR text reference
// (e.g. "14 CFR § 91.3"); clicking a source expands the exact retrieved passage
// in-app so the user can verify the claim without leaving the page. Open state
// is owned by the parent so an inline [n] chip can drive it (`focus`): when the
// user clicks a chip, we scroll its pill into view and pulse it.
function Sources({ sources, openN, setOpenN, focus }) {
  const pillRefs = useRef(new Map())
  const [pulseN, setPulseN] = useState(null)

  useEffect(() => {
    if (!focus) return
    const el = pillRefs.current.get(focus.n)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    setPulseN(focus.n)
    const t = setTimeout(() => setPulseN(null), 1100)
    return () => clearTimeout(t)
    // `focus` carries a bumped tick, so repeat clicks on the same chip re-fire.
  }, [focus])

  if (!sources.length) return null
  const open = sources.find((c) => c.n === openN) || null
  const setRef = (n) => (el) => {
    if (el) pillRefs.current.set(n, el)
    else pillRefs.current.delete(n)
  }
  return (
    <div className="sources">
      <span className="sources-label">Sources</span>
      {sources.map((c) => {
        const isOpen = openN === c.n
        const label = c.cfr_citation || c.source
        const cls = `source-pill${c.text ? ' toggle' : ''}${isOpen ? ' open' : ''}${
          pulseN === c.n ? ' pulse' : ''
        }`
        // With no retrieved text there is nothing to expand — render a plain
        // text reference (still a jump target for its inline chip).
        if (!c.text) {
          return (
            <span key={c.n} ref={setRef(c.n)} className={cls} title={`chunk ${c.chunk_index}`}>
              [{c.n}] {label}
            </span>
          )
        }
        return (
          <button
            key={c.n}
            ref={setRef(c.n)}
            type="button"
            className={cls}
            aria-expanded={isOpen}
            onClick={() => setOpenN(isOpen ? null : c.n)}
            title="Show the exact retrieved passage"
          >
            [{c.n}] {label}{' '}
            <span className="caret" aria-hidden="true">{isOpen ? '▾' : '▸'}</span>
          </button>
        )
      })}
      {open && open.text && (
        // Keyed by n so switching sources remounts it and replays the entrance.
        <blockquote key={open.n} className="passage">
          <span className="passage-head">
            [{open.n}] {open.cfr_citation || open.source}
          </span>
          {open.quote && (
            <p className="passage-quote">
              <mark title="Verified supporting quote from this source">{`“${open.quote}”`}</mark>
            </p>
          )}
          {open.text}
        </blockquote>
      )}
    </div>
  )
}

// Copy the raw answer text to the clipboard, with a brief "Copied" confirmation.
// Falls back to a hidden-textarea + execCommand when the async Clipboard API is
// unavailable (older browsers or a non-secure context).
function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef(null)

  useEffect(() => () => clearTimeout(timerRef.current), [])

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try {
        document.execCommand('copy')
      } catch {
        /* nothing more we can do; leave `copied` false */
        document.body.removeChild(ta)
        return
      }
      document.body.removeChild(ta)
    }
    setCopied(true)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setCopied(false), 1500)
  }

  return (
    <button
      type="button"
      className={`copy-btn${copied ? ' copied' : ''}`}
      onClick={copy}
      title="Copy answer to clipboard"
      aria-label={copied ? 'Answer copied' : 'Copy answer'}
    >
      {copied ? '✓ Copied' : '⧉ Copy'}
    </button>
  )
}

function Message({ m, onRetry, sending }) {
  if (m.role === 'user') {
    return (
      <div className="row user">
        <div className="bubble">{m.text}</div>
      </div>
    )
  }

  if (m.role === 'error') {
    const notice = m.tone === 'notice'
    return (
      <div className={`row ${notice ? 'notice' : 'error'}`}>
        <div className="avatar" aria-hidden="true">{notice ? 'i' : '!'}</div>
        <div className="bubble">
          <span>{m.text}</span>
          {m.retry && (
            <div className="error-actions">
              <button className="link" onClick={() => onRetry(m.retry)} disabled={sending}>
                Retry
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  return <AssistantMessage m={m} />
}

// An assistant answer plus its sources. Holds the open-source state so a click
// on an inline [n] chip (via onCite) and a click on a Sources pill drive the
// same expansion — clicking a chip also scrolls to and pulses its pill.
function AssistantMessage({ m }) {
  const [openN, setOpenN] = useState(null)
  const [focus, setFocus] = useState(null)
  // Stable (empty deps) so the memoized <Markdown> doesn't re-parse on open.
  const onCite = useCallback((n) => {
    setOpenN(n)
    setFocus((f) => ({ n, tick: (f?.tick || 0) + 1 }))
  }, [])

  // While the answer is still streaming and no text has landed yet, show the
  // typing indicator — this placeholder message replaces the old standalone one.
  if (m.streaming && !m.text) {
    return (
      <div className="row assistant">
        <div className="avatar" aria-hidden="true">A</div>
        <div className="bubble typing-bubble" aria-label="Searching the corpus and thinking">
          <span className="typing">
            <span></span>
            <span></span>
            <span></span>
          </span>
        </div>
      </div>
    )
  }

  // Streaming with partial text — or an interrupted partial that never reached
  // the verified `done` payload (Stop / drop / error mid-stream). Either way the
  // citation set is unknown, so render the text with [n] markers neutral (via
  // `pending`) and hold the sources / verification / metrics footer: routing an
  // unverified partial through the verifier with an empty citation set would
  // falsely flag its real citations as hallucinated. The blinking caret (a CSS
  // ::after) shows only while text is actively arriving.
  if (m.streaming || m.interrupted) {
    return (
      <div className="row assistant">
        <div className="avatar" aria-hidden="true">A</div>
        <div className="bubble">
          <div className={`answer${m.streaming ? ' streaming' : ''}`}>
            <Markdown text={m.text} citations={m.citations} pending />
          </div>
        </div>
      </div>
    )
  }

  const sources = m.citations || []
  // Verify the [n] markers in the final answer against the sources the backend
  // actually returned, so the UI can vouch for what it renders (and flag any
  // reference with no matching source instead of quietly trusting it).
  const verify = verifyCitations(m.text, sources)
  // Backend-neutralized [?] markers have no number but still count as unresolved.
  const badCount = verify.invalid.length + (verify.unresolved || 0)
  // "verified" only when the backend checked each [n] against a verbatim quote
  // from the cited passage (#11); otherwise the check is mere resolution.
  const verified = m.meta?.citations_verified
  const nCite = verify.valid.length
  const okText = verified
    ? `✓ ${nCite} citation${nCite === 1 ? '' : 's'} verified against the cited passage`
    : `✓ ${nCite} citation${nCite === 1 ? '' : 's'} resolve${nCite === 1 ? 's' : ''} to a retrieved source`
  return (
    <div className="row assistant">
      <div className="avatar" aria-hidden="true">A</div>
      <div className="bubble">
        <div className="answer">
          {/* Pass m.citations (stable ref) not `sources` (fresh []) so memo holds. */}
          <Markdown text={m.text} citations={m.citations} onCite={onCite} />
        </div>
        <Sources sources={sources} openN={openN} setOpenN={setOpenN} focus={focus} />
        {/* A non-abstained answer that cited nothing is ungrounded — say so
            plainly rather than rendering it as a normal, source-backed answer. */}
        {sources.length === 0 && !m.abstained && (
          <div className="verify">No sources cited for this answer.</div>
        )}
        {verify.total > 0 && (
          <div className={`verify ${badCount ? 'warn' : 'ok'}`}>
            {badCount === 0
              ? okText
              : `⚠ ${badCount} unverified reference${badCount === 1 ? '' : 's'}${
                  verify.invalid.length
                    ? ` (${verify.invalid.map((n) => `[${n}]`).join(', ')})`
                    : ''
                } — no matching source`}
          </div>
        )}
        {m.meta && <Metrics meta={m.meta} />}
        <div className="msg-actions">
          <CopyButton text={m.text} />
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [status, setStatus] = useState('idle') // 'idle' | 'sending'
  // Seeded from the attribute the inline script in index.html resolved before
  // paint (saved choice, else OS preference). Until the user toggles, we don't
  // persist — so the app keeps following the OS on each visit.
  const [theme, setTheme] = useState(
    () => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'),
  )
  const abortRef = useRef(null)
  const taRef = useRef(null)
  const bottomRef = useRef(null)

  const sending = status === 'sending'
  const isEmpty = messages.length === 0

  // Reflect the current theme onto <html> so the CSS variables switch.
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  function toggleTheme() {
    setTheme((t) => {
      const next = t === 'dark' ? 'light' : 'dark'
      try {
        localStorage.setItem('theme', next)
      } catch {
        /* private mode / storage disabled — the toggle still works this session */
      }
      return next
    })
  }

  // Keep the latest message (or the typing indicator) in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, status])

  function resetComposerHeight() {
    if (taRef.current) taRef.current.style.height = 'auto'
  }

  function pushError(err, question, didTimeout) {
    let text
    let tone = 'error'
    if (err?.name === 'AbortError') {
      if (didTimeout) {
        text = `The request timed out after ${REQUEST_TIMEOUT_MS / 1000}s — the backend may be slow or stuck.`
      } else {
        text = 'Request stopped.'
        tone = 'notice'
      }
    } else if (err instanceof TypeError) {
      // fetch() rejects with a TypeError when the server can't be reached.
      text =
        "Couldn't reach the backend. Make sure it's running on http://127.0.0.1:5000 — " +
        'in another terminal run:  cd backend && python app.py'
    } else {
      text = err?.message || 'Something went wrong while contacting the server.'
    }
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: 'error', tone, text, retry: tone === 'notice' ? null : question },
    ])
  }

  async function send(raw) {
    const q = (typeof raw === 'string' ? raw : input).trim()
    if (!q || sending) return

    // Add the user turn and an empty streaming assistant placeholder up front, so
    // deltas can flow straight into it (and it doubles as the typing indicator).
    const assistantId = nextId()
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: 'user', text: q },
      { id: assistantId, role: 'assistant', text: '', citations: [], abstained: false, meta: null, streaming: true },
    ])
    setInput('')
    resetComposerHeight()
    setStatus('sending')

    const controller = new AbortController()
    abortRef.current = controller
    let didTimeout = false
    const timer = setTimeout(() => {
      didTimeout = true
      controller.abort()
    }, REQUEST_TIMEOUT_MS)

    // Update only our streaming assistant message (matched by its stable id).
    const patch = (fn) =>
      setMessages((prev) => prev.map((mm) => (mm.id === assistantId ? fn(mm) : mm)))

    let gotText = false
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ message: q }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const body = await res.text().catch(() => '')
        const detail = body.replace(/\s+/g, ' ').trim().slice(0, 200)
        throw new Error(
          `Server responded ${res.status} ${res.statusText}${detail ? ` — ${detail}` : ''}`,
        )
      }
      if (!res.body) throw new Error('The server returned a response with no body to stream.')

      // Parse the SSE stream: events are separated by a blank line, and each frame
      // carries a single `data: {json}` line in our protocol.
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let done = null

      for (;;) {
        const { value, done: streamDone } = await reader.read()
        if (streamDone) break
        buf += decoder.decode(value, { stream: true })
        let sep
        while ((sep = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, sep)
          buf = buf.slice(sep + 2)
          const dataLine = frame.split('\n').find((l) => l.startsWith('data:'))
          if (!dataLine) continue
          let evt
          try {
            evt = JSON.parse(dataLine.slice(5).trim())
          } catch {
            continue // ignore an unparseable frame rather than dropping the stream
          }
          if (evt.type === 'delta') {
            gotText = true
            patch((mm) => ({ ...mm, text: mm.text + (evt.text || '') }))
          } else if (evt.type === 'done') {
            done = evt
          } else if (evt.type === 'error') {
            throw new Error(evt.message || 'The server hit an error while generating the answer.')
          }
        }
      }

      if (!done) throw new Error('The stream ended before the answer was complete.')
      const reply = (done.reply ?? '').toString().trim()
      if (!reply) throw new Error('The server returned an empty answer.')
      // Settle on the authoritative payload: replace the provisional streamed text
      // (which differs only where a [n] was neutralized to [?]) and attach the
      // verified citations + meta.
      patch((mm) => ({
        ...mm,
        text: reply,
        citations: Array.isArray(done.citations) ? done.citations : [],
        // The gate's out-of-scope refusal (abstained) legitimately carries no
        // citations; a normal answer with none is ungrounded — distinguish them.
        abstained: done.abstained === true,
        meta: done.meta || null,
        streaming: false,
      }))
    } catch (err) {
      // Interrupted mid-stream: keep any partial answer already on screen so it
      // doesn't vanish, but mark it `interrupted` (not a normal settled answer) —
      // it never got the verified `done` payload, so it must NOT be run through
      // citation verification (which would flag its real [n] as hallucinated).
      // Otherwise drop the empty placeholder.
      if (gotText) {
        patch((mm) => ({ ...mm, streaming: false, interrupted: true }))
      } else {
        setMessages((prev) => prev.filter((mm) => mm.id !== assistantId))
      }
      pushError(err, q, didTimeout)
    } finally {
      clearTimeout(timer)
      abortRef.current = null
      setStatus('idle')
      requestAnimationFrame(() => taRef.current?.focus())
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  function clearChat() {
    if (sending) return
    setMessages([])
    requestAnimationFrame(() => taRef.current?.focus())
  }

  function onSubmit(e) {
    e.preventDefault()
    send()
  }

  function onKeyDown(e) {
    // Enter sends; Shift+Enter inserts a newline. Ignore Enter mid-IME-composition
    // so committing a Korean/Japanese/Chinese candidate doesn't fire a send.
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      send()
    }
  }

  function onInput(e) {
    setInput(e.target.value)
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>14 CFR RAG Chat</h1>
          <p className="subtitle">
            Grounded answers over Title 14 CFR (FAA aviation regulations) — every claim cited to
            its <code>14 CFR §</code> source, or the system abstains.
          </p>
        </div>
        <div className="header-actions">
          <button
            className="icon-btn"
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <button
            className="ghost"
            onClick={clearChat}
            disabled={isEmpty || sending}
            title="Clear the conversation"
          >
            Clear
          </button>
        </div>
      </header>

      <main className="messages" role="log" aria-live="polite" aria-relevant="additions">
        {isEmpty ? (
          <div className="empty">
            <div className="empty-emoji" aria-hidden="true">✈️</div>
            <p>Ask a question about the FAA regulations in 14 CFR. Try one of these:</p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex} className="chip" onClick={() => send(ex)} disabled={sending}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <Message key={m.id} m={m} onRetry={send} sending={sending} />
          ))
        )}
        {/* The streaming assistant placeholder (added on send) is the typing
            indicator now — it shows the dots until the first token lands. */}
        <div ref={bottomRef} />
      </main>

      <div className="composer">
        <form onSubmit={onSubmit}>
          <textarea
            ref={taRef}
            value={input}
            onChange={onInput}
            onKeyDown={onKeyDown}
            placeholder="Ask about 14 CFR…  (Enter to send · Shift+Enter for a new line)"
            rows={1}
            autoFocus
            aria-label="Your question"
          />
          {sending ? (
            <button type="button" className="btn stop" onClick={stop} title="Stop the request">
              <span className="spinner" aria-hidden="true" />
              Stop
            </button>
          ) : (
            <button type="submit" className="btn send" disabled={!input.trim()} title="Send">
              Send
            </button>
          )}
        </form>
        <p className="hint">
          Answers are limited to the indexed documents — the model is instructed to say when it
          doesn't know.
        </p>
      </div>
    </div>
  )
}
