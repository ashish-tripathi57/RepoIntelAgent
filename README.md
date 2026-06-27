# RepoIntel Multi-Agent System

A multi-agent orchestration framework built with LangGraph and LangChain. It uses the Model Context Protocol (MCP) alongside Google Gemini and Tavily Web Search to perform autonomous repository analysis and generate strategic business intelligence. Supports automatic model failover, LangSmith tracing, and step-by-step streaming.

## Architecture

The system uses a state-graph topology with two specialized agents:

**Research Agent** — Acts as the data aggregator. Translates natural language queries into deterministic API tool calls via `bind_tools()`. Bridges two data sources:
- Native MCP subprocess: spawns the RepoIntel server as an STDIO-bound child process, avoiding raw network exposure.
- Tavily web search: scans community blogs, CVE trackers, and technology forums.

**Advisory Agent** — Pure-reasoning synthesis node. Consumes collected JSON and web content, then applies structured prompt analysis to deliver a GO / NO-GO advisory report with risk matrices and mitigations.

Graph execution follows a strict linear path: `START -> Research -> Advisory -> END`.

## Key Features

- **Deterministic MCP abstraction**: Extracts repository health metrics via the official `mcp.ClientSession` STDIO protocol rather than HTTP.
- **State passthrough**: Zero external databases; all state flows through a LangGraph `TypedDict`.
- **Model failover**: Automatic retry with `gemini-2.5-flash` when the primary model (`gemini-3.1-pro-preview`) times out. Both agents exit gracefully if all models fail.
- **LangSmith tracing**: Full observability into LLM calls, tool invocations, and graph transitions. Enable by setting `LANGCHAIN_API_KEY`.
- **Streaming execution**: Graph uses `astream(stream_mode="updates")` to log each node's output in real time.

## Prerequisites

1. Python 3.11 or newer.
2. A local copy of the RepoIntel backend, runnable via `python main.py --mode mcp`.
3. Active API keys for Google Gemini and Tavily.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY="your_gemini_api_key"
GEMINI_MODEL="gemini-3.1-pro-preview"
GEMINI_FALLBACK_MODEL="gemini-2.5-flash"
TAVILY_API_KEY="your_tavily_api_key"
GITHUB_TOKEN="your_github_token"

# Optional — LangSmith observability
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="RepoIntelAgent"
```

## Usage

```bash
python agent.py
```

**Sample output:**

```
Enter query: What are the security vulnerabilities in langchain-ai/langchain?

[Research Agent] Analyzing query...
[Research Agent] Calling gemini-3.1-pro-preview... succeeded.
[Research Agent] Tool tavily_search returned 4120 characters.
[Graph] Node 'research' completed.

[Advisory Agent] Synthesizing final report...
[Advisory Agent] Calling gemini-3.1-pro-preview... succeeded.
[Graph] Node 'advise' completed.

--- FINAL REPORT ---
GO / NO-GO: CONDITIONAL
Confidence: Medium
```

**Failover behavior:**

```
[Research Agent] Primary model timed out. Falling back to gemini-2.5-flash...
[Research Agent] gemini-2.5-flash succeeded.
```

## Testing

This project follows Test-Driven Development. All agent interactions are tested via mocked network stubs using `pytest` and `unittest.mock`, with a strict >90% coverage threshold.

```bash
pytest --cov=. --cov-fail-under=90
```

**Test coverage:**

| File | Scope |
|------|-------|
| `test_research_agent.py` | Tool binding, node schema, primary model, failover, graceful exit |
| `test_advisory_agent.py` | LLM formatting, failover, error handling |
| `test_graph.py` | Graph topology, `astream` support, end-to-end streaming |
