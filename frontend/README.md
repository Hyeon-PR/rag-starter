# frontend

React + Vite chat UI for the **14 CFR** RAG backend. Dependency-free (React only) — no extra packages to install.

**Features**
- Chat bubbles (user / assistant), avatars, full-height sticky composer, light + dark themes
- **Markdown rendering** of answers — headings, **bold**/*italic*, `code`, bullet/numbered lists, blockquotes, rules — via a tiny in-repo renderer (`src/markdown.jsx`) that builds React elements directly (no `dangerouslySetInnerHTML`, XSS-safe by construction; no third-party dependency)
- Inline `[n]` citation badges (hover for source file + chunk) and a Sources line under each answer
- **Citation-token verification** — every `[n]` in the final answer is checked against the sources the backend returned; a status line reads "✓ N citations verified" or flags any reference with no matching source (also rendered as a distinct `n?` badge) so a hallucinated marker is never passed off as real
- Robust error handling: backend-unreachable hint, HTTP/JSON/empty-reply errors, 90s timeout, **Stop** to abort, **Retry** on failed requests
- Loading "typing" indicator, auto-scroll, empty-state with clickable example questions
- Textarea composer: Enter to send, Shift+Enter for a newline, IME-safe (won't send mid-composition)

## Setup

```bash
cd frontend
npm install
npm run dev
# Running on http://localhost:5173
```

Make sure the backend is running on port 5000 in another terminal.
