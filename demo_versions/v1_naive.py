"""
VERSION 1 — NAIVE.
All four tools ARE available (GitHub MCP tools + Tavily web search), but a weak,
neutral prompt and a single model call mean the model never thinks to use the web
tool for this kind of query. So it confidently recommends GO having done ZERO
security research — even though the capability was sitting right there.

The scariest failure isn't a missing capability. It's a capability that's present
and ignored.
"""
from _shared import mcp_tools, research_llm, tavily_tool, _sanitize_tool_args, run


async def research_node(state: dict) -> dict:
    async with mcp_tools() as (_session, github_tools):
        # Tavily IS available — the model COULD research security. The weak prompt
        # below just never nudges it to, so for this query it doesn't bother.
        tools = github_tools + [tavily_tool]
        llm = research_llm().bind_tools(tools)
        prompt = f"Gather data to answer this query:\n{state['user_query']}\nUse the tools available to you."
        msg = await llm.ainvoke(prompt)

        tmap = {t.name: t for t in tools}
        github, web = [], []
        print(f"[Single-shot] tools available: {[t.name for t in tools]}")
        print(f"[Single-shot] model chose {len(msg.tool_calls)} tool(s): {[t['name'] for t in msg.tool_calls]}")
        for tc in msg.tool_calls:
            args = _sanitize_tool_args(tc["name"], tc.get("args", {}))
            if tc["name"] in tmap:
                res = str(await tmap[tc["name"]].ainvoke(args))
                (web if "tavily" in tc["name"] else github).append(res)

        if not web:
            print("[Single-shot] NOTE: tavily_search was AVAILABLE but the model never called it "
                  "-> ZERO security research, yet it will still produce a confident GO.")
        return {"github_data": str(github), "web_research": str(web)}


if __name__ == "__main__":
    run("VERSION 1 — NAIVE  (web tool available but ignored -> confident GO, zero security research)", research_node)
