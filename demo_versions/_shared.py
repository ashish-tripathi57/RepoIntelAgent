"""
Shared plumbing for the 4 demo versions of the Research Agent.

Each version (v1..v4) changes ONLY the research strategy. Everything else — the
MCP connection, the GitHub tools, Tavily, the graph, and the (local Qwen) Advisory
Agent — lives here and is reused. Run a version with, e.g.:

    python demo_versions/v1_naive.py
"""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tavily import TavilyClient

# Make the parent package importable so we can reuse your Advisory Agent (Qwen).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.advisory_agent import advisory_node  # noqa: E402

load_dotenv()

# RepoIntel MCP server lives next to RepoIntelAgent: ../RepoIntel/main.py
REPO_INTEL_MAIN = os.path.join(os.path.dirname(ROOT), "RepoIntel", "main.py")


class AgentState(TypedDict):
    user_query: str
    github_data: str
    web_research: str
    final_report: str


class RepoArgs(BaseModel):
    owner: str
    repo: str


class IssueArgs(BaseModel):
    owner: str
    repo: str
    limit: int = 5


def tavily_search(query: str) -> str:
    """Search the web for security advisories, CVEs, news, and community sentiment."""
    if "TAVILY_API_KEY" not in os.environ:
        return "Tavily key missing"
    return str(TavilyClient().search(query=query))


tavily_tool = StructuredTool.from_function(
    func=tavily_search,
    name="tavily_search",
    description="Search the web for security vulnerabilities, CVEs, advisories, news, and sentiment about a repository.",
)


def _sanitize_tool_args(tool_name: str, args: dict) -> dict:
    """Fix hallucinated parameter names (e.g. 'repository' -> 'repo')."""
    args = dict(args)
    if tool_name in ("get_repo_info", "list_open_issues", "analyze_repo_health"):
        if "repo" not in args:
            for k in ("arg", "repository", "name", "repo_name"):
                if k in args:
                    args["repo"] = args.pop(k)
                    break
        if "owner" not in args:
            for k in ("org", "organization", "user", "owner_name"):
                if k in args:
                    args["owner"] = args.pop(k)
                    break
    return args


def research_llm():
    """The research model (cloud, tool-calling). Defaults to a stable Gemini flash."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("RESEARCH_PRIMARY_MODEL", "gemini-2.5-flash"),
        temperature=0,
    )


@asynccontextmanager
async def mcp_tools():
    """Open the RepoIntel MCP server (stdio) and yield (session, github_tools)."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[REPO_INTEL_MAIN, "--mode", "mcp"],
        env={**os.environ},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = await session.list_tools()
            print("[Research] Discovered MCP tools:", [t.name for t in discovered.tools])

            async def _gri(owner: str, repo: str) -> str:
                return str((await session.call_tool("get_repo_info", {"owner": owner, "repo": repo})).content)

            async def _loi(owner: str, repo: str, limit: int = 5) -> str:
                return str((await session.call_tool("list_open_issues", {"owner": owner, "repo": repo, "limit": limit})).content)

            async def _arh(owner: str, repo: str) -> str:
                return str((await session.call_tool("analyze_repo_health", {"owner": owner, "repo": repo})).content)

            github_tools = [
                StructuredTool.from_function(coroutine=_gri, name="get_repo_info", description="Fetch GitHub repository info via MCP.", args_schema=RepoArgs),
                StructuredTool.from_function(coroutine=_loi, name="list_open_issues", description="List recent open issues via MCP.", args_schema=IssueArgs),
                StructuredTool.from_function(coroutine=_arh, name="analyze_repo_health", description="Compute repo health metrics via MCP.", args_schema=RepoArgs),
            ]
            yield session, github_tools


def _as_text(x) -> str:
    """Advisory models return content as a string (Ollama/Qwen) OR a list of content
    blocks (Gemini: [{'type':'text','text':...}]). Coerce either to clean text."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in x)
    return str(x)


def run(version_label: str, research_node):
    """Build the research -> advise graph and run it on a user query."""
    builder = StateGraph(AgentState)
    builder.add_node("research", research_node)
    builder.add_node("advise", advisory_node)
    builder.add_edge(START, "research")
    builder.add_edge("research", "advise")
    builder.add_edge("advise", END)
    graph = builder.compile()

    print("=" * 64)
    print(f"  {version_label}")
    print("=" * 64)
    query = input("Enter query: ")

    async def _go():
        state = {"user_query": query, "github_data": "", "web_research": "", "final_report": ""}
        async for step in graph.astream(state, stream_mode="updates"):
            for node, out in step.items():
                print(f"\n[Graph] Node '{node}' completed.")
                state.update(out)
        print("\n--- FINAL REPORT ---\n")
        print(_as_text(state["final_report"]))

    asyncio.run(_go())
