# Frontend integration

A short note for the platform team consuming this API. Covers the bits
that aren't obvious from the OpenAPI schema alone — mostly: **how to
render math content** in the API responses.

---

## API contract

The API exposes two endpoints the platform actually needs:

| Endpoint | Use |
|----------|-----|
| `POST /quiz/generate` | The main call — give it a topic, get a quiz back. |
| `POST /retrieve`      | Debug-only — see what the retriever pulls, no LLM. |

Full schemas at `http://<host>:8000/docs` (auto-generated Swagger UI).

### Request example (math)

```json
POST /quiz/generate
{
  "topic": "Fonction Logarithme",
  "language": "fr",
  "subject": "MATHEMATICS",
  "school_phase": "HIGH",
  "count": 5
}
```

**Recommended filters:**
- Always pass `language` (required).
- Pass `subject` when known — narrows the retrieval pool meaningfully.
- Pass `school_phase` (PRIMARY / MIDDLE / HIGH) whenever you know the
  student's grade. This is the single biggest production-realism lever
  and removes most cross-level retrieval noise. **For math, this is
  especially important** — without it, primary-school content can leak
  into high-school requests.
- `levels` (plural, e.g., `["HIGH_SCHOOL_2ND_GRADE_LETTRES"]`) is
  available for finer-grained filtering when needed.

### Response shape

```json
{
  "topic": "Fonction Logarithme",
  "language": "fr",
  "subject": "MATHEMATICS",
  "level": null,
  "questions": [
    {
      "question_type": "MULTIPLE_CHOICE",
      "question_text": "La forme algébrique de \\(z = 4e^{i\\pi/3}\\) est :",
      "choices": [
        "\\(2 + 2i\\sqrt{3}\\)",
        "\\(2\\sqrt{3} + 2i\\)",
        "\\(4 + 4i\\)"
      ],
      "correct_answers": ["\\(2 + 2i\\sqrt{3}\\)"],
      "multiple_correct_answers": false,
      "explanation": "On a \\(z = 4(\\cos(\\pi/3) + i\\sin(\\pi/3)) = ...\\).",
      "difficulty": "medium"
    }
    // ... more questions ...
  ]
}
```

### Error response

```json
{ "detail": "Generation failed: <reason>" }
```

Possible reasons (visible to you):
- `Generation failed after 3 attempts. Last error: ...` — LLM
  consistently failed validation; very rare.
- `unsupported_language` — input language not in {en, fr, ar}.
- Any 5xx from the LLM provider — usually transient, retry the request.

---

## Rendering math content (THE important integration step)

The `question_text`, `choices`, `correct_answers`, and `explanation`
fields **may contain LaTeX** in the math subject. Examples:

```
\(\frac{1}{2}\)              ← inline math
\(\sqrt{x+1}\)
\(e^{i\pi/4}\)
```

These are standard LaTeX. They are **NOT pre-rendered to HTML or
images** by the backend — your frontend has to render them.

### Integration with MathJax (simplest)

Add these two `<script>` tags to your page:

```html
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

That's it. MathJax auto-detects `\(...\)` and `\[...\]` blocks on the
page and renders them. So rendering the API response is just:

```jsx
// React example — but the principle is the same in any framework
function Question({ q }) {
  return (
    <div>
      <p>{q.question_text}</p>
      <ul>
        {q.choices.map(c => <li key={c}>{c}</li>)}
      </ul>
    </div>
  );
}

// After React paints, call MathJax to re-typeset:
useEffect(() => {
  window.MathJax?.typesetPromise?.();
}, [questions]);
```

### Integration with KaTeX (faster, slightly less feature coverage)

[katex.org](https://katex.org/) — comparable docs. Use this if MathJax
is too heavy for your bundle. KaTeX handles every LaTeX pattern this
backend currently emits.

### What if the LaTeX is broken?

It shouldn't be — the backend has a [LaTeX validity check](../src/generation/latex_validity.py)
that catches broken markup and retries the LLM before the response is
returned. But if something gets through:

- MathJax will display a small red error indicator over that single
  expression. The rest of the question still renders.
- The page is **never** corrupted by a broken math block (the
  validator's primary job is preventing exactly that).

If you see a recurring pattern of bad LaTeX, ping the AI team — it
means a new failure mode the validator doesn't yet cover.

---

## Authentication

**API key.** Send it on every call except `/health`:

```
X-API-Key: <your-key>
```

`Authorization: Bearer <your-key>` is accepted as an equivalent.

Keys are configured server-side via the `API_KEYS` environment variable
(comma-separated, so several callers can hold distinct keys). When
`API_KEYS` is unset the API runs **unauthenticated** — that's the local
development default, and the server logs a warning at startup saying so.
Never run a reachable deployment that way.

Failure modes:

| Status | Meaning |
|--------|---------|
| `401` | Missing or invalid key |
| `429` | Rate limit exceeded — honour the `Retry-After` header (seconds) |

`/health` is deliberately left open so container healthchecks and load
balancers work without credentials. It reveals nothing but liveness.

### Rate limiting

Each caller gets `RATE_LIMIT_PER_MINUTE` requests (default 30) across
`/retrieve` and `/quiz/generate`, over a rolling 60-second window.
Callers are identified by API key when present, otherwise by client IP.
The budget is counted **per server process** — with multiple workers the
effective ceiling is higher, so treat this as a cost guard rather than a
precise quota.

### Correlation IDs

Every response carries an `X-Request-ID` header. Send your own to have it
preserved end to end; otherwise the server generates one.

**Log it against your user record.** When an error body says
`(request_id=abc123...)`, that ID is what lets the AI team find the full
traceback in the server log. Error responses are deliberately vague —
they never include exception text, file paths, or corpus internals — so
the ID is the only thread back to the cause.

---

## Logging & support requests

Every call lands in `logs/runs.jsonl` on the API host, indexed by
timestamp. If a teacher complains about a bad quiz, find the entry
matching their request time and forward it to the AI team — the entry
includes the full request, full LLM response, and timings.

Example:

```bash
# Find calls from this afternoon for Arabic
jq -c 'select(.timestamp > "2026-05-19T13:00")
       | select(.request.language == "ar")' logs/runs.jsonl
```

---

## Quick checklist for going live

- [ ] MathJax (or KaTeX) integrated and re-typesetting after each API
      response renders.
- [ ] `school_phase` is being passed on every `/quiz/generate` call.
- [ ] Auth layer in front of the API.
- [ ] Error handling: surface HTTP 502 / 500 as "try again in a moment"
      to the teacher; surface HTTP 400 as "please refine your topic."
- [ ] Logging: tee API requests on your side so they're correlated with
      user accounts (backend's `runs.jsonl` doesn't know who the
      teacher is).
