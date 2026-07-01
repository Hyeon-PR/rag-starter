import { useEffect, useRef, useState } from 'react'

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

  // assistant
  const sources = m.citations || []
  // Verify the [n] markers in the final answer against the sources the backend
  // actually returned, so the UI can vouch for what it renders (and flag any
  // reference with no matching source instead of quietly trusting it).
  const verify = verifyCitations(m.text, sources)
  return (
    <div className="row assistant">
      <div className="avatar" aria-hidden="true">A</div>
      <div className="bubble">
        <div className="answer">
          {/* Pass m.citations (stable ref) not `sources` (fresh []) so memo holds. */}
          <Markdown text={m.text} citations={m.citations} />
        </div>
        {sources.length > 0 && (
          <div className="sources">
            <span className="sources-label">Sources</span>
            {sources.map((c) => (
              <span key={c.n} className="source-pill" title={`chunk ${c.chunk_index}`}>
                [{c.n}] {c.source}
              </span>
            ))}
          </div>
        )}
        {verify.total > 0 && (
          <div className={`verify ${verify.invalid.length ? 'warn' : 'ok'}`}>
            {verify.invalid.length === 0
              ? `✓ ${verify.valid.length} citation${verify.valid.length === 1 ? '' : 's'} verified against sources`
              : `⚠ ${verify.invalid.length} unverified reference${
                  verify.invalid.length === 1 ? '' : 's'
                } (${verify.invalid.map((n) => `[${n}]`).join(', ')}) — no matching source`}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [status, setStatus] = useState('idle') // 'idle' | 'sending'
  const abortRef = useRef(null)
  const taRef = useRef(null)
  const bottomRef = useRef(null)

  const sending = status === 'sending'
  const isEmpty = messages.length === 0

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

    setMessages((prev) => [...prev, { id: nextId(), role: 'user', text: q }])
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

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

      let data
      try {
        data = await res.json()
      } catch {
        throw new Error('The server returned a response that was not valid JSON.')
      }

      const reply = (data?.reply ?? '').toString().trim()
      if (!reply) throw new Error('The server returned an empty answer.')

      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          text: reply,
          citations: Array.isArray(data.citations) ? data.citations : [],
        },
      ])
    } catch (err) {
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
        <button
          className="ghost"
          onClick={clearChat}
          disabled={isEmpty || sending}
          title="Clear the conversation"
        >
          Clear
        </button>
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

        {sending && (
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
        )}
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
