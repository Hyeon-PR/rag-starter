# frontend

React + Vite chat UI for the RAG backend. Dependency-free (React only) — no extra packages to install.

**Features**
- Chat bubbles (user / assistant), avatars, full-height sticky composer, light + dark themes
- Inline `[n]` citation badges (hover for source file + chunk) and a Sources line under each answer
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
