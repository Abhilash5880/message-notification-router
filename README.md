# Message Notification Router

A context-aware notification routing system that classifies incoming messages as **Notify**, **Digest**, or **Mute** using deterministic policy rules, historical user behavior, temporal context, safety analysis, and optional multimodal LLM reasoning.

The project is designed around a simple question:

> **Does this message deserve the user's attention right now?**

Rather than treating notification routing as a basic text-classification problem, the router considers the recipient, sender history, prior interactions, urgency, safety signals, group context, business relationships, notification load, and available media.

## Results

On the development evaluation set:

- **96.7% action accuracy**
- **96.7% message-type accuracy**
- **96.7% joint accuracy**
- **29 / 30 samples classified correctly**
- Deterministic offline fallback with no API dependency

The remaining failure involved a textless urgent voice message when running without ASR, motivating the optional multimodal extraction pipeline.

## Architecture

```text
Incoming Message
       │
       ▼
┌─────────────────────┐
│ Data / Context      │
│ User, sender, group │
│ business + history  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Media Extraction    │
│ OCR / Vision / ASR  │
│ + local fallback    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Safety Analysis     │
│ Scam / OTP / links  │
│ prompt injection    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Evidence Retrieval  │
│ Historical messages│
│ + user behavior     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Routing Policy      │
│ Notify / Digest /   │
│ Mute                │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Confidence + Reason │
└─────────────────────┘
```

## Key Features

### Context-Aware Routing

Routing decisions incorporate signals such as:

- message urgency and deadlines
- direct requests and mentions
- sender history
- group membership and mute state
- user notification load
- business relationships
- previous opens, replies, dismissals, and reports
- user opt-in / opt-out behavior

### Temporal Evidence Retrieval

Historical evidence is strictly **time-causal**.

For a message at time `T`, the router only considers information available before `T`. Future interactions cannot leak into the current prediction.

Historical messages are ranked using recipient/entity alignment, textual similarity, recency, and behavioral evidence.

### Safety-First Policy Engine

Clear safety signals are handled locally before optional model reasoning.

Detection includes patterns related to:

- suspicious payment requests
- OTPs and login codes
- credential requests
- suspicious links/domains
- impersonation
- spam-like forwarding behavior
- prompt-injection attempts

Untrusted message content is treated strictly as data and cannot override routing instructions.

### Multimodal Support

The router supports media-aware classification for:

- images
- screenshots
- posters
- voice notes

Optional OCR, vision, and ASR extraction can enrich the message representation.

Extracted media summaries are cached by content hash to avoid unnecessary repeated processing.

If media extraction is unavailable, the system falls back to deterministic metadata/caption-based routing with appropriately reduced confidence.

### Hybrid LLM + Algorithmic Routing

The system does **not** require an LLM call for every message.

Deterministic rules handle clear cases locally. An optional provider adapter can be used for ambiguous cases requiring additional semantic reasoning.

This provides:

- predictable behavior
- lower API usage
- offline operation
- graceful degradation
- provider independence

### Confidence Calibration

Prediction confidence is derived from factors including:

- evidence strength
- source trust
- agreement between routing signals
- media extraction quality
- conflicting contextual signals

Each prediction contains both a routing decision and a concise explanation.

## Project Structure

```text
code/
├── evaluation/
│   └── main.py
├── router/
│   ├── __init__.py
│   ├── confidence.py
│   ├── data.py
│   ├── engine.py
│   ├── llm.py
│   ├── media.py
│   ├── output.py
│   ├── policy.py
│   ├── retrieval.py
│   └── safety.py
├── tests/
│   └── test_router.py
└── main.py
```

### Core Modules

| Module | Responsibility |
|---|---|
| `engine.py` | Coordinates the complete routing pipeline |
| `data.py` | Loads data and constructs time-causal context |
| `policy.py` | Deterministic notification routing logic |
| `safety.py` | Detects unsafe, suspicious, and adversarial content |
| `retrieval.py` | Retrieves relevant historical evidence |
| `media.py` | Handles image/audio extraction and caching |
| `llm.py` | Optional provider-backed semantic reasoning |
| `confidence.py` | Calibrates prediction confidence |
| `output.py` | Validates and serializes predictions |

## Running the Router

The core implementation uses Python and can operate without network access.

```bash
python code/main.py
```

## Running Tests

```bash
python -m unittest discover -s code/tests -v
```

The test suite covers areas including:

- temporal evidence validity
- future-data leakage prevention
- output contract validation
- safety handling
- prompt-injection resistance
- media-derived safety signals

## Development Evaluation

```bash
python code/evaluation/main.py --json
```

The evaluation pipeline reports action accuracy, message-type accuracy, joint accuracy, and failure diagnostics.

Evaluation data is kept separate from production inference and is never used as a direct label lookup.

## Optional LLM Integration

The router can operate completely offline, but optional provider-backed reasoning can be configured through environment variables.

```env
ROUTER_LLM_URL=
ROUTER_LLM_API_KEY=
ROUTER_LLM_MODEL=

ROUTER_MEDIA_EXTRACTOR_URL=
ROUTER_MEDIA_EXTRACTOR_API_KEY=
```

A direct OpenAI-backed multimodal path is also supported when configured:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_VISION_MODEL=
OPENAI_TRANSCRIPTION_MODEL=
OPENAI_REASONING_MODEL=
```

**Never commit API keys or `.env` files to the repository.**

When no provider or network connection is available, the deterministic routing pipeline remains operational.

## Design Principles

The project follows four main principles:

1. **Safety before semantic reasoning**
2. **No future-data leakage**
3. **LLMs enhance decisions rather than being required for them**
4. **Every routing decision should remain explainable**

## Future Work

Potential extensions include:

- Android notification-listener integration
- WhatsApp notification routing
- personalized online preference learning
- local on-device language models
- improved offline OCR and ASR
- learned ranking for historical evidence retrieval
- notification priority scheduling
- cross-application notification aggregation

## Background

This project originated as a solution to a message-notification routing challenge and was subsequently structured as a standalone notification intelligence system.

The current implementation focuses on the routing engine itself, with future work aimed at integrating it with real-world notification sources such as Android applications and messaging platforms.
