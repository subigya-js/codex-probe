# CodexProbe

CodexProbe is a configurable local reverse proxy that records LLM calls made by Codex CLI. It sits between Codex and an OpenAI-compatible backend, forwards requests and responses, preserves live Server-Sent Events (SSE) streaming, reconstructs streamed responses for logs, and stores one ordered JSONL file per recorder session.

```text
Codex CLI <-> CodexProbe <-> OpenAI
                         <-> Ollama / Qwen
```

The same proxy implementation is used for every backend. Switching backends requires configuration changes only.

## What CodexProbe records

For each inference request, CodexProbe records:

- session identity and call order;
- UTC start time and latency;
- backend name, base URL, and wire API;
- complete request method, path, and JSON body;
- response status and whether it streamed;
- complete response body, reconstructed from SSE when needed;
- a safe error description when no response is produced.

CodexProbe records `/responses` and `/chat/completions` inference requests. Supporting requests such as `/models` are forwarded but are not counted as LLM calls.

CodexProbe does not execute tools. Codex executes tools locally and includes tool results in later model requests, which CodexProbe can then record.

## Requirements

- Python 3.11 or newer
- Codex CLI for the Codex demonstrations
- Ollama for the local Qwen demonstration
- An OpenAI Platform API key for the OpenAI demonstration

The verified environment was:

- Python 3.13.5
- Codex CLI 0.153.0
- Ollama 0.18.0
- `qwen2.5-coder:7b`

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

Install development dependencies when running the tests:

```bash
pip install -e ".[dev]"
```

Runtime and development dependencies are pinned in `pyproject.toml`.

## Public API

```python
from codex_probe import ProxyRecorder


config = {
    "backend": {
        "name": "ollama-qwen2.5-coder",
        "base_url": "http://127.0.0.1:11434/v1",
        "wire_api": "responses",
        "auth": {
            "mode": "none",
        },
    },
    "listen": {
        "host": "127.0.0.1",
        "port": 0,
    },
    "log_dir": ".codex-probe/logs",
}

recorder = ProxyRecorder(config)
endpoint = recorder.start()

print(f"Configure Codex to use: {endpoint}")

try:
    input("Press Enter to stop CodexProbe.\n")
finally:
    calls = recorder.stop()

print(f"Recorded {len(calls)} LLM call(s).")
```

`start()` runs the local proxy in a background thread and returns its HTTP endpoint. `stop()` shuts down the server, writes the session JSONL file, and returns an ordered `list[dict]` of recorded calls.

A `ProxyRecorder` instance is single-use: call `start()` once and `stop()` once.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `backend.name` | yes | Non-sensitive backend identity stored in logs |
| `backend.base_url` | yes | Upstream OpenAI-compatible base URL |
| `backend.wire_api` | yes | `responses` or `chat` |
| `backend.auth.mode` | yes | `none`, `environment`, or `forward` |
| `backend.auth.environment_variable` | conditional | Required only for `environment` authentication |
| `listen.host` | no | Local bind address; defaults to `127.0.0.1` |
| `listen.port` | no | Local port; `0` asks the OS to choose a free port |
| `log_dir` | no | Session log directory; defaults to `.codex-probe/logs` |

For safety, the listener accepts loopback addresses only. Plain HTTP backend URLs are accepted only for loopback hosts; remote backends must use HTTPS.

### Authentication modes

`none`
: Removes any incoming authorization header and sends no backend credential. This is used for local Ollama.

`environment`
: Reads a token from the configured environment variable and sends it to the backend as a bearer token. The OpenAI example uses `OPENAI_API_KEY`.

`forward`
: Forwards the incoming authorization header to the backend. The header is never recorded.

## Run with Ollama and Qwen

Download and serve the model:

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

In another terminal, start CodexProbe:

```bash
source .venv/bin/activate
python examples/run_proxy.py examples/ollama-qwen.json
```

The script prints an endpoint such as:

```text
http://127.0.0.1:54650
```

Use the printed endpoint as the custom Codex provider `base_url`. The port changes when `listen.port` is `0`.

Example non-interactive Codex invocation, replacing `<endpoint>` with the printed URL:

```bash
codex -s read-only -a never exec \
  --ignore-user-config \
  --ephemeral \
  -m qwen2.5-coder:7b \
  -c 'model_provider="codexprobe_qwen"' \
  -c 'model_providers.codexprobe_qwen.name="CodexProbe Qwen"' \
  -c 'model_providers.codexprobe_qwen.base_url="<endpoint>"' \
  -c 'model_providers.codexprobe_qwen.wire_api="responses"' \
  -c 'model_providers.codexprobe_qwen.requires_openai_auth=false' \
  -c 'model_providers.codexprobe_qwen.request_max_retries=0' \
  -c 'model_providers.codexprobe_qwen.stream_max_retries=0' \
  'Reply with exactly CODEXPROBE_QWEN_OK. Do not use any tools.'
```

Press Enter in the proxy terminal after Codex finishes. This writes the session log.

## Run with OpenAI

Create an OpenAI Platform API key. Do not place it in a configuration file. Set it only in the terminal that starts CodexProbe:

```bash
read -s "OPENAI_API_KEY?Paste OpenAI API key: "
echo
export OPENAI_API_KEY
```

Start CodexProbe:

```bash
python examples/run_proxy.py examples/openai.json
```

Example Codex invocation, replacing `<endpoint>` with the printed URL:

```bash
codex -s read-only -a never exec \
  --ignore-user-config \
  --ephemeral \
  -m gpt-5.6-luna \
  -c 'model_provider="codexprobe_openai"' \
  -c 'model_reasoning_effort="none"' \
  -c 'model_reasoning_summary="none"' \
  -c 'model_providers.codexprobe_openai.name="CodexProbe OpenAI"' \
  -c 'model_providers.codexprobe_openai.base_url="<endpoint>"' \
  -c 'model_providers.codexprobe_openai.wire_api="responses"' \
  -c 'model_providers.codexprobe_openai.requires_openai_auth=false' \
  -c 'model_providers.codexprobe_openai.request_max_retries=0' \
  -c 'model_providers.codexprobe_openai.stream_max_retries=0' \
  'Reply with exactly CODEXPROBE_OPENAI_OK. Do not use any tools.'
```

Here, Codex does not send a credential to the local proxy. CodexProbe reads `OPENAI_API_KEY` and authenticates the upstream request. Remove the variable after the demonstration:

```bash
unset OPENAI_API_KEY
```

## Streaming behavior

For SSE responses, CodexProbe handles each backend chunk in two ways:

1. It immediately writes the chunk to the Codex client, preserving live streaming.
2. It feeds the same chunk into an incremental SSE parser and response reassembler.

Responses API streams use their terminal response event as the complete recorded response. Chat Completions streams are rebuilt from their deltas, including text, refusal content, and function tool calls.

Compressed backend responses are decompressed before forwarding and recording. Stale `Content-Encoding` and `Content-Length` headers are removed so clients do not attempt to decompress the body twice.

If reconstruction fails, already received bytes are still forwarded. The call is recorded with a safe reconstruction error instead of a partial response object.

## Session logs

Each `ProxyRecorder` instance represents one recording session. Calls are numbered when their requests begin, so concurrent calls retain request-start order even if they finish in a different order.

Stopping the recorder creates:

```text
<log_dir>/<session_id>.jsonl
```

Each line is one schema-versioned call:

```json
{
  "schema_version": 1,
  "session_id": "b02656ddfffd4429a0b6bd1764a5346c",
  "call_index": 1,
  "started_at": "2026-09-25T14:00:00.000000+00:00",
  "latency_ms": 1234.5,
  "backend": {
    "name": "ollama-qwen2.5-coder",
    "base_url": "http://127.0.0.1:11434/v1",
    "wire_api": "responses"
  },
  "request": {
    "method": "POST",
    "path": "/responses",
    "body": {
      "model": "qwen2.5-coder:7b",
      "input": "Example prompt"
    }
  },
  "response": {
    "status": 200,
    "streamed": true,
    "body": {
      "object": "response",
      "status": "completed",
      "model": "qwen2.5-coder:7b"
    }
  },
  "error": null
}
```

Exactly one of `response` and `error` is non-null.

## Verified results

The following end-to-end behavior was verified on September 25, 2026:

| Route | Result |
| --- | --- |
| Direct Ollama `/v1/responses` request | Passed |
| `curl -> CodexProbe -> Ollama/Qwen` streaming | Passed |
| `Codex CLI -> CodexProbe -> Ollama/Qwen` streaming | Passed |
| Qwen backend identity and model recorded | Passed |
| Direct OpenAI `/v1/responses` request | Passed |
| `curl -> CodexProbe -> OpenAI` | Passed |
| `Codex CLI -> CodexProbe -> OpenAI` streaming | Passed |
| OpenAI backend identity and model recorded | Passed |
| Compressed OpenAI response reconstruction | Passed |

The verified OpenAI Codex session recorded one streamed `/responses` call with HTTP status 200, backend `openai`, and response model `gpt-5.6-luna`. The verified Qwen Codex session recorded one streamed `/responses` call with HTTP status 200, backend `ollama-qwen2.5-coder`, and response model `qwen2.5-coder:7b`.

### Qwen tool-calling observation

One controlled Codex run asked `qwen2.5-coder:7b` to read `pyproject.toml` using a shell tool. The recorded request contained 16 tool definitions, including function, namespace, and web-search tool types. The backend returned HTTP 200, but its response contained only a normal message and no structured function-call item. Codex therefore executed no tool and made no follow-up LLM request.

This result is intentionally scoped to the tested combination of Codex CLI 0.153.0, Ollama 0.18.0, `qwen2.5-coder:7b`, the Responses API, and that prompt. It does not prove that Qwen or Ollama can never call tools. Ollama documents general function-calling support, so the observed limitation may involve the model, the mixed Codex tool schema, the prompt, or their interaction.

### Model discovery warning

Codex 0.153.0 attempted `/models?client_version=0.153.0` requests and expected a response containing a `models` field. Both tested OpenAI-compatible backends returned the standard `object` and `data` model-list shape. Codex printed a warning, used its configured or fallback model metadata, and completed inference successfully. CodexProbe forwarded these discovery requests but correctly excluded them from LLM call numbering.

## Security and privacy

CodexProbe never persists HTTP headers. Therefore authorization headers, API keys, cookies, and backend credentials are excluded from recordings.

Request and response bodies are intentionally complete. They may contain:

- prompts and conversation history;
- source code and file contents;
- tool definitions and tool results;
- command output and error messages.

Treat every generated JSONL file as sensitive. The default `.codex-probe/` directory and `*.jsonl` files are excluded from Git by `.gitignore`.

CodexProbe binds only to loopback interfaces. It is not designed to be exposed as a public network service.

## Limitations

- Calls are held in memory and written when `stop()` runs. A process crash or forced termination before shutdown can lose the current session.
- Request bodies are buffered in memory. The implementation targets JSON LLM requests, not arbitrary large uploads.
- The proxy supports HTTP OpenAI-compatible endpoints, not WebSocket model transport.
- Current Codex custom providers use the Responses API. Chat Completions support is retained for compatible non-Codex clients and assignment coverage.
- Session boundaries follow `ProxyRecorder` lifecycle boundaries, not individual prompts. All inference requests between `start()` and `stop()` belong to one session.
- Exact backend model behavior, especially tool calling, depends on the selected model and serving implementation.

## Tests

Run the complete suite:

```bash
pytest
```

The suite covers:

- configuration validation and safe defaults;
- authentication behavior without credential persistence;
- request and response passthrough;
- backend failures and redirects;
- concurrent call ordering;
- SSE parsing across arbitrary network chunks;
- Responses and Chat Completions reconstruction;
- tool-call delta reconstruction;
- compressed response handling;
- JSONL schema and session behavior;
- public `ProxyRecorder` lifecycle and startup failure;
- example configuration validation;
- exclusion of non-inference `/models` requests.

## Project structure

```text
codex-probe/
├── examples/
│   ├── ollama-qwen.json
│   ├── openai.json
│   └── run_proxy.py
├── src/codex_probe/
│   ├── __init__.py
│   ├── config.py
│   ├── proxy.py
│   ├── recorder.py
│   ├── recording.py
│   └── sse.py
├── tests/
│   ├── test_config.py
│   ├── test_examples.py
│   ├── test_proxy.py
│   ├── test_recorder.py
│   ├── test_recording.py
│   └── test_sse.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## References

- [Codex advanced configuration](https://developers.openai.com/codex/config-advanced)
- [Codex configuration reference](https://developers.openai.com/codex/config-reference)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
