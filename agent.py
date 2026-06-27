from typing import TypedDict
import os
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from agents import research_node, advisory_node

load_dotenv()

# LangSmith tracing — set LANGCHAIN_API_KEY in .env to enable
if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
    print(f"[LangSmith] Tracing enabled for project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
else:
    print("[LangSmith] Tracing disabled — set LANGCHAIN_API_KEY in .env to enable.")

class AgentState(TypedDict):
    user_query: str
    github_data: str
    web_research: str
    final_report: str

def setup_graph():
    builder = StateGraph(AgentState)
    # The graph compilation abstracts coroutine checks so async nodes operate flawlessly transparent to other units.
    builder.add_node("research", research_node)
    builder.add_node("advise", advisory_node)
    
    builder.add_edge(START, "research")
    builder.add_edge("research", "advise")
    builder.add_edge("advise", END)
    
    return builder.compile()

async def run_agent(query: str):
    graph = setup_graph()
    initial_state = {
        "user_query": query,
        "github_data": "",
        "web_research": "",
        "final_report": ""
    }
    # Stream through each node for step-by-step visibility
    final_state = initial_state
    async for step in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in step.items():
            print(f"\n{'='*60}")
            print(f"[Graph] Node '{node_name}' completed.")
            for key, value in node_output.items():
                preview = str(value)[:200]
                print(f"  → {key}: {preview}{'...' if len(str(value)) > 200 else ''}")
            print(f"{'='*60}")
            final_state.update(node_output)
    return final_state["final_report"]

async def main():  # pragma: no cover
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please configure .env secrets before testing.")
        exit(1)
        
    query = input("Enter query for RepoIntel MAS: ")
    print("\n--- BEGIN EXECUTION ---")
    report = await run_agent(query)
    print("\n--- FINAL REPORT Handoff ---")
    print(report)

    # --- Optional: hand the verdict off to n8n (set WEBHOOK_URL to enable) ---------
    webhook = os.environ.get("WEBHOOK_URL")
    if webhook:
        import json as _json
        import urllib.request as _url
        # advisory output may be a string (Qwen) or a list of blocks (Gemini) — flatten it
        if isinstance(report, list):
            report_text = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in report)
        else:
            report_text = str(report)
        verdict = "NO-GO" if "NO-GO" in report_text.upper() else ("GO" if "GO" in report_text.upper() else "REVIEW")
        body = _json.dumps({"query": query, "verdict": verdict, "report": report_text}).encode()
        try:
            _url.urlopen(_url.Request(webhook, data=body, headers={"Content-Type": "application/json"}), timeout=30)
            print(f"[n8n] verdict '{verdict}' sent to webhook")
        except Exception as e:
            print(f"[n8n] webhook post failed: {e}")

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
