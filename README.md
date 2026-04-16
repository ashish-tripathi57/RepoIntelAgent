# 🕵️‍♂️ RepoIntel Multi-Agent System

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-%E2%9C%93-brightgreen)
![LangGraph](https://img.shields.io/badge/LangGraph-%E2%9C%93-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-%3E90%25-success)
![Gemini](https://img.shields.io/badge/Model-Gemini%203.1%20Pro-purple)

A highly resilient, multi-agent orchestration framework built with **LangGraph** and **LangChain**. It seamlessly leverages the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) alongside Google's **Gemini 3.1 Pro** and **Tavily Web Search** to perform autonomous, deep-dive repository analysis and formulate strategic business intelligence. Features **automatic model failover**, **LangSmith tracing**, and **step-by-step streaming** for full observability.

---

## 📖 Table of Contents
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration (`.env`)](#configuration)
- [Usage Executions](#usage-executions)
- [Testing & Test-Driven Development (TDD)](#testing--tdd)

---

## 🧠 Architecture

The system utilizes a state-graph topology to enforce role-based execution using a two-agent strategy:

1. **Research Agent (Node)**: Acts as the data aggregater. It dynamically translates natural language queries into deterministic API tool calls using `bind_tools()`. It bridges:
   - **Native MCP Python Subprocessing**: Spawns the RepoIntel `main.py` explicitly as an STDIO-bound child process asynchronously without raw network exposure.
   - **Tavily Web Agent**: Digging across community blogs, CVE trackers, and technology forums.
   
2. **Advisory Agent (Node)**: Pure-reasoning synthesis node. Takes the collected JSON structures and web strings, applying strict prompt analysis to deliver a GO / NO-GO advisory report, complete with risk matrices and mitigations.

## ✨ Key Features
- **Deterministic MCP Abstraction**: Extracts backend repository health metrics cleanly executing the official `mcp.ClientSession` STDIO parameters rather than naive HTTP.
- **Micro-Graph Execution**: Enforces `START -> Research -> Advisory -> END` linear constraints, preventing agent hallucinations.
- **State Passthrough Architecture**: Uses zero external databases; state is managed cleanly over LangGraph `TypedDict`.
- **Model Failover Chain**: Automatic retry with a faster fallback model (`gemini-2.5-flash`) when the primary (`gemini-3.1-pro-preview`) times out after 30 seconds. Both agents exit gracefully with collected data if both models fail.
- **LangSmith Tracing**: Full observability into LLM calls, tool invocations, and graph transitions via LangSmith. Enable by setting `LANGCHAIN_API_KEY` in `.env`.
- **Step-by-Step Streaming**: Graph execution uses `astream(stream_mode="updates")` to log each node's output as it completes, giving real-time visibility into the pipeline.

---

## ⚙️ Prerequisites
Before running the orchestrator, ensure you have:
1. **Python 3.11** or newer.
2. A local copy of the **RepoIntel** backend logic mapping to execute over `python main.py --mode mcp`. No concurrent HTTP server must be active!
3. Active API keys for both Google Gemini and Tavily.

---

## 🚀 Installation & Setup

1. **Clone the project** and navigate to the directory.
2. **Create a virtual environment (Recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install the project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔐 Configuration

Create a `.env` file in the project root to house your secrets. A template is defined below:

```env
# Required AI APIs
GEMINI_API_KEY="your_actual_gemini_api_key_here"
GEMINI_MODEL="gemini-3.1-pro-preview"
GEMINI_FALLBACK_MODEL="gemini-2.5-flash"

# Required Search API
TAVILY_API_KEY="your_tavily_search_api_key_here"

# Pathing Configuration for the Local MCP Wrapper
GITHUB_TOKEN="your_github_token_here"

# LangSmith Observability (optional)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="RepoIntelAgent"
```

---

## 🖥️ Usage Executions

Start the LangGraph orchestration flow via the terminal. The CLI handles interactive queries:

```bash
python agent.py
```

### Sample Workflow
```text
Enter query for RepoIntel MAS: What are the security vulnerabilities surrounding langchain-ai/langchain?

--- BEGIN EXECUTION ---
[LangSmith] Tracing enabled for project: RepoIntelAgent

[Research Agent] Analyzing query: What are the security vulnerabilities...
[Research Agent] Calling gemini-3.1-pro-preview...
[Research Agent] gemini-3.1-pro-preview succeeded.
[Research Agent] Calling Tool: tavily_search with args: {'query': 'langchain security cve 2026 vulnerabilities'}
[Research Agent] Tool tavily_search returned 4120 characters.

============================================================
[Graph] Node 'research' completed.
  → github_data: [{'repo': 'langchain-ai/langchain', ...}...
  → web_research: [{'title': 'CVE-2026-...', ...}...
============================================================

[Advisory Agent] Synthesizing final report...
[Advisory Agent] Calling gemini-3.1-pro-preview...
[Advisory Agent] gemini-3.1-pro-preview succeeded.
[Advisory Agent] Final recommendation complete.

============================================================
[Graph] Node 'advise' completed.
  → final_report: ## Recommendation: CONDITIONAL...
============================================================

--- FINAL REPORT Handoff ---
GO / NO-GO: CONDITIONAL
Confidence Score: Medium
Risk Factors: ...
```

### Failover Example
When the primary model is slow or unavailable:
```text
[Research Agent] Calling gemini-3.1-pro-preview...
[Research Agent] Primary model timed out. Falling back to gemini-2.5-flash...
[Research Agent] gemini-2.5-flash succeeded.
```

---

## 🧪 Testing & TDD

This project strictly adheres to **Test-Driven Development (TDD)** practices. All core graph traversals and agent interactions are mapped via mocked network stubs using `pytest` and `unittest.mock`. 

We enforce a strict **>90%** coverage threshold across the repository.

To run the test suite and validate threshold metrics:
```bash
pytest --cov=. --cov-fail-under=90
```

### Known Coverage Map
- `test_research_agent.py`: Validates tool binding, node schema, primary model success, failover to fallback, and graceful exit when both models fail.
- `test_advisory_agent.py`: Validates LLM formatting, failover to fallback model, and graceful error message on total failure.
- `test_graph.py`: Validates graph topology, `astream` support, and end-to-end streaming through both nodes.
