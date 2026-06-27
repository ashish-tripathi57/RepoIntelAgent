import os
import sys
import json
import asyncio
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from tavily import TavilyClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class RepoArgs(BaseModel):
    owner: str
    repo: str

class IssueArgs(BaseModel):
    owner: str
    repo: str
    limit: int = 5

def tavily_search(query: str) -> str:
    """Search for recent news, security advisories, CVEs, and sentiment."""
    if "TAVILY_API_KEY" not in os.environ: return "Tavily key missing"
    tavily = TavilyClient()
    return str(tavily.search(query=query))

tavily_tool = StructuredTool.from_function(func=tavily_search, name="tavily_search", description="Search for recent news, security advisories, CVEs, and sentiment.")

def _sanitize_tool_args(tool_name: str, args: dict) -> dict:
    """Fix hallucinated parameter names from LLMs that don't match the tool schema."""
    return dict(args)  # <-- DEMO BREAK: guardrail disabled. DELETE THIS LINE to restore it.
    args = dict(args)
    if tool_name in ("get_repo_info", "list_open_issues", "analyze_repo_health"):
        # Fix common hallucinations: 'arg', 'repository', 'name' → 'repo'
        if "repo" not in args:
            for wrong_key in ("arg", "repository", "name", "repo_name"):
                if wrong_key in args:
                    args["repo"] = args.pop(wrong_key)
                    print(f"[Research Agent] Auto-corrected param '{wrong_key}' → 'repo'")
                    break
        if "owner" not in args:
            for wrong_key in ("org", "organization", "user", "owner_name"):
                if wrong_key in args:
                    args["owner"] = args.pop(wrong_key)
                    print(f"[Research Agent] Auto-corrected param '{wrong_key}' → 'owner'")
                    break
    return args


async def research_node(state: dict) -> dict:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["/Users/ashishtripathi/Workspaces/RepoIntel/main.py", "--mode", "mcp"],
        env={**os.environ}
    )

    print(f"\n[Research Agent] Analyzing query: {state['user_query']}")
    collected_github = []
    collected_web = []

    # 1. Start MCP session directly launching subprocess from binary path
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 2 & 3. Discover available tools and prove connectivity
            mcp_tools = await session.list_tools()
            print("[Research Agent] Discovered MCP Tools:", [t.name for t in mcp_tools.tools])
            
            # Form bridging mappings that leverage real session scope
            async def _get_repo_info(owner: str, repo: str) -> str:
                res = await session.call_tool("get_repo_info", arguments={"owner": owner, "repo": repo})
                return str(res.content)
                
            async def _list_open_issues(owner: str, repo: str, limit: int = 5) -> str:
                res = await session.call_tool("list_open_issues", arguments={"owner": owner, "repo": repo, "limit": limit})
                return str(res.content)
                
            async def _analyze_repo_health(owner: str, repo: str) -> str:
                res = await session.call_tool("analyze_repo_health", arguments={"owner": owner, "repo": repo})
                return str(res.content)

            tools = [
                StructuredTool.from_function(coroutine=_get_repo_info, name="get_repo_info", description="Fetch basic GitHub repository information via MCP.", args_schema=RepoArgs),
                StructuredTool.from_function(coroutine=_list_open_issues, name="list_open_issues", description="Fetch recent open issues from a GitHub repository via MCP.", args_schema=IssueArgs),
                StructuredTool.from_function(coroutine=_analyze_repo_health, name="analyze_repo_health", description="Compute health metrics for a GitHub repository via MCP.", args_schema=RepoArgs),
                tavily_tool
            ]

            primary_model = os.environ.get("RESEARCH_PRIMARY_MODEL", "gemini-3.1-pro-preview")
            fallback_model = os.environ.get("RESEARCH_FALLBACK_MODEL", "gemini-2.5-flash")
            query = f"""You are a research analyst. Gather data to answer this query thoroughly:

{state['user_query']}

Use the tools available to you to collect both repository-level data from GitHub and external information from the web. A thorough analysis typically requires data from multiple sources."""

            # Failover chain: primary (Gemini) → fallback (Gemini) → graceful exit
            try:
                print(f"[Research Agent] Calling {primary_model}...")
                llm = ChatGoogleGenerativeAI(model=primary_model).bind_tools(tools)
                msg = await asyncio.wait_for(llm.ainvoke(query), timeout=300)
                print(f"[Research Agent] {primary_model} succeeded.")
            except (asyncio.TimeoutError, Exception) as e:
                print(f"[Research Agent] Primary model failed ({type(e).__name__}). Falling back to {fallback_model}...")
                try:
                    llm = ChatGoogleGenerativeAI(model=fallback_model).bind_tools(tools)
                    msg = await asyncio.wait_for(llm.ainvoke(query), timeout=300)
                    print(f"[Research Agent] {fallback_model} succeeded.")
                except (asyncio.TimeoutError, Exception) as e2:
                    print(f"[Research Agent] Both models failed. Primary: {type(e).__name__}, Fallback: {type(e2).__name__}")
                    return {
                        "github_data": str(collected_github),
                        "web_research": str(collected_web)
                    }
            
            for tool_call in msg.tool_calls:
                args = _sanitize_tool_args(tool_call['name'], tool_call['args'])
                print(f"[Research Agent] Calling Tool: {tool_call['name']} with args: {args}")
                matched = [t for t in tools if t.name == tool_call['name']]
                if not matched:
                    print(f"[Research Agent] Unknown tool '{tool_call['name']}', skipping.")
                    continue
                try:
                    result = await matched[0].ainvoke(args)
                    print(f"[Research Agent] Tool {tool_call['name']} returned {len(str(result))} characters.")
                    if "tavily" in tool_call['name']: collected_web.append(result)
                    else: collected_github.append(result)
                except Exception as tool_err:
                    print(f"[Research Agent] Tool {tool_call['name']} failed: {tool_err}. Skipping.")
    
    # Session closes autonomously here due to exiting context blocks
    return {
        "github_data": str(collected_github),
        "web_research": str(collected_web)
    }
