# Message Notification Router

Run from the repository root:

```powershell
python code/main.py
```

The router uses only the Python standard library and writes a validated
`dataset/output.csv`. It is deterministic when no network configuration exists.

## Optional integrations

No provider is assumed. Configure an adapter endpoint only when available:

```text
ROUTER_LLM_URL=https://your-adapter.example/route
ROUTER_LLM_API_KEY=...
ROUTER_LLM_MODEL=optional-model-name
ROUTER_MEDIA_EXTRACTOR_URL=https://your-adapter.example/media
ROUTER_MEDIA_EXTRACTOR_API_KEY=...
```

The LLM adapter receives a JSON object containing a task, policy, structured
facts, and expected schema; it must return either that decision object or a
`result` object with `action`, `message_type`, `reason`, and `confidence`.
The media adapter returns a `result` object with optional `text`, `summary`, and
`quality`. API failure always falls back to deterministic routing.

## Direct OpenAI option

When `OPENAI_API_KEY` is present, no custom adapter is required. Images use the
Responses API for OCR/visual summaries, voice notes use the transcription API,
and ambiguous routing can use the Responses API. Override defaults with
`OPENAI_BASE_URL`, `OPENAI_VISION_MODEL`, `OPENAI_TRANSCRIPTION_MODEL`, and
`OPENAI_REASONING_MODEL`. Clear local safety cases are never sent to the
reasoning model; no key or network still yields deterministic output.

## Development regression

Samples are evaluation-only and are never loaded by production routing:

```powershell
python code/evaluation/main.py
python code/evaluation/main.py --json
```

The report gives action/type accuracy and every failure with a generalizable
diagnosis. Do not turn individual sample IDs or answers into production rules.
