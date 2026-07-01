# frontend

React + Vite chat UI for the **14 CFR** RAG backend. Dependency-free (React only) — no extra packages to install.

**Features**
- Chat bubbles (user / assistant), avatars, full-height sticky composer, light + dark themes
- **Markdown rendering** of answers — headings, **bold**/*italic*, `code`, bullet/numbered lists, blockquotes, rules — via a tiny in-repo renderer (`src/markdown.jsx`) that builds React elements directly (no `dangerouslySetInnerHTML`, XSS-safe by construction; no third-party dependency)
- Inline `[n]` citation badges (hover for source + chunk) and a **Sources** line under each answer — each source links out to the official **eCFR** section (new tab, `rel="noopener"`) so you can jump straight to the material, and expands in-app (`▸ source text`) to show the exact retrieved passage that grounded the answer
- **Citation-token verification** — every `[n]` in the final answer is checked against the sources the backend returned; a status line reads "✓ N citations resolve to a retrieved source" (resolution, not entailment — it does not check that the passage supports the claim) and flags any unresolved reference (rendered as a distinct `n?` / `[?]` badge) so a hallucinated marker is never passed off as real. A non-abstained answer that cited nothing is marked "No sources cited."
- **Per-answer cost + latency** — input/output token counts, the estimated **`$` cost per answer**, and per-stage timing (retrieval / llm / total, from the backend `meta`) shown in a compact line under each answer; the model name is in its tooltip
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
