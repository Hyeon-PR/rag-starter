import { useEffect, useRef, useState } from 'react'

// Monotonic message ids (stable React keys without depending on array index).
let idSeq = 0
const nextId = () => ++idSeq

const EXAMPLES = [
  'What was the cause of the Apollo 1 fire?',
  'Which Apollo missions landed on the Moon?',
  'Compare the moonwalk durations of Apollo 11 and Apollo 17.',
  'List Apollo missions that used the Saturn V rocket.',
  'What is the Artemis program?',
]

// Anthropic + retrieval can take a while; give it room but don't hang forever.
const REQUEST_TIMEOUT_MS = 90_000

// Render an answer, turning inline [n] markers into citation badges.
// Unknown numbers (not in the citation set) are left as plain text so we never
// pretend a hallucinated marker is a real source.
function AnswerText({ text, citations }) {
  const byNum = new Map((citations || []).map((c) => [c.n, c]))
  const parts = []
  const re = /\[(\d+)\]/g
  let last = 0
  let key = 0
  let m
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    const n = Number(m[1])
    const c = byNum.get(n)
    if (c) {
      parts.push(
        <sup key={`c${key++}`} className="cite" title={`${c.source} · chunk ${c.chunk_index}`}>
          {n}
        </sup>,
      )
    } else {
      parts.push(m[0])
    }
    last = re.lastIndex
  }
  if (last < text.length) parts.push(text.slice(last))
  return <>{parts}</>
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

  // assistant
  const sources = m.citations || []
  return (
    <div className="row assistant">
      <div className="avatar" aria-hidden="true">A</div>
      <div className="bubble">
        <div className="answer">
          <AnswerText text={m.text} citations={sources} />
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
          <h1>RAG Chat</h1>
          <p className="subtitle">
            Grounded answers over the Apollo corpus — every claim cited to a source file.
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
            <div className="empty-emoji" aria-hidden="true">🚀</div>
            <p>Ask a question about the Apollo program. Try one of these:</p>
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
            placeholder="Ask about Apollo missions…  (Enter to send · Shift+Enter for a new line)"
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
