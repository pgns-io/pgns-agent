# pgns-agent

Express.js for A2A agents.

[![PyPI](https://img.shields.io/pypi/v/pgns-agent)](https://pypi.org/project/pgns-agent/)
[![Python](https://img.shields.io/pypi/pyversions/pgns-agent)](https://pypi.org/project/pgns-agent/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Wrap any agent function in a production-ready [A2A](https://google.github.io/A2A/)-compatible server, powered by [pgns](https://pgns.io).

```python
from pgns_agent import AgentServer

agent = AgentServer("my-agent", "Echoes input")

@agent.on_task
async def handle(task):
    return {"echo": task.input}

agent.listen(3000)
```

## Quickstart

**Install:**

```bash
pip install pgns-agent
```

**Configure** (optional — omit for local dev mode):

```bash
export PGNS_API_KEY="your-api-key"
```

**Run:**

```bash
python agent.py
```

Your agent is now live. Verify the agent card at [http://localhost:3000/.well-known/agent.json](http://localhost:3000/.well-known/agent.json).

## Features

- **A2A protocol compliance** — serves the standard Agent Card at `/.well-known/agent.json`
- **Auto-provisioning** — registers an agent card and roost on the pgns relay at startup
- **Sync + async modes** — handlers are always async, but the server can run standalone or embedded
- **SSE streaming** — stream task progress updates to callers via Server-Sent Events
- **Task lifecycle** — `submitted → working → completed/failed` with full status tracking
- **Input-required flow** — suspend a handler and request additional input from the caller
- **Pipeline observability** — AgentServer records execution stages automatically across every agent in a pipeline
- **Local dev mode** — run without a pgns API key for local development and testing

## Three entry points

**Standalone server** — start a self-contained HTTP server:

```python
agent.listen(3000)
```

**Mount into an existing app** — get a Starlette/ASGI application to compose with FastAPI, Starlette, or any ASGI framework:

```python
app = agent.app()
```

**Serverless handler** — get the raw ASGI callable for Lambda, Cloud Functions, etc.:

```python
handler = agent.handler()
```

## Framework adapters

Adapters let you wire existing agent frameworks into pgns-agent with zero glue code.

### LangChain / LangGraph

```bash
pip install pgns-agent-langchain
```

```python
from pgns_agent import AgentServer
from pgns_agent.adapters import LangChainAdapter

agent = AgentServer("my-langchain-agent", "Runs a LangChain chain")
agent.use(LangChainAdapter(my_runnable))
agent.listen(3000)
```

### OpenAI Agents SDK

```bash
pip install pgns-agent-openai
```

### Claude Agent SDK

```bash
pip install pgns-agent-claude
```

### CrewAI

```bash
pip install pgns-agent-crewai
```

## Pipeline observability

AgentServer records execution stages automatically. No instrumentation code required.

Every handler invocation produces a trace stage: agent name, start time, duration, and success or failure. When one agent calls another via `agent.send()`, the trace propagates automatically through the pipeline. By the time the final agent completes, the trace contains the full execution history — which agents ran, in what order, how long each took, and where failures occurred.

The trace is visible in the pgns dashboard Pipeline Trace Viewer, linked to the correlation ID for the pipeline run.

**Annotate stages for richer traces** (optional):

```python
@agent.on_task
async def handle(task):
    result = run_my_model(task.input)

    # Optional: annotate the trace stage with output context
    task.trace.annotate(
        output_summary=f"Generated {len(result['text'])} chars",
        metadata={"model": "gpt-4o", "tokens": result["usage"]["total_tokens"]},
    )

    return result
```

Without annotation, the trace records timing and status. Annotation adds context — useful for debugging latency or inspecting intermediate outputs across a multi-agent pipeline.

**Before (manual trace management):**

```python
# ~200 lines of boilerplate across a 5-agent pipeline:
# begin_trace, begin_stage, complete_stage, fail_stage,
# append_stage, inject_trace, extract_trace — in every handler
stage = begin_stage("writer")
try:
    result = run_writer(task.input)
    stage = complete_stage(stage, output_summary=result["summary"])
except Exception as e:
    stage = fail_stage(stage, error=str(e))
    raise
finally:
    trace = append_stage(trace, stage)
    payload = inject_trace({"result": result}, trace)
```

**After:**

```python
@agent.on_task
async def handle(task):
    return run_writer(task.input)  # trace recorded automatically
```

## Testing

Use the built-in test client for unit testing — no server required:

```python
client = agent.test_client()
resp = client.send_task({"text": "hello"})
assert resp.status == "completed"
assert resp.result == {"echo": {"text": "hello"}}
```

## Documentation

Full documentation is available at [docs.pgns.io/libraries/pgns-agent](https://docs.pgns.io/libraries/pgns-agent).

- [API Reference](https://docs.pgns.io/libraries/pgns-agent/api)
- [GitHub](https://github.com/pgns-io/pgns-agent)

## License

Apache 2.0
