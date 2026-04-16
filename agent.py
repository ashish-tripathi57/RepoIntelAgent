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

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
