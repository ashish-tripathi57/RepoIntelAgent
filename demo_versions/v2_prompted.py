"""
VERSION 2 — STRONG PROMPT.
GitHub + Tavily tools, single model call, model decides. A firm "you MUST research
security" instruction makes the model *usually* call Tavily — but it's still the
model's choice in one shot, so it is not guaranteed. (Run it a few times: it'll
mostly comply, occasionally not. That unreliability is the point.)
"""
from _shared import mcp_tools, research_llm, tavily_tool, _sanitize_tool_args, run

STRONG = """You are a due-diligence analyst. For EVERY repository in the query you MUST:
1) call get_repo_info AND analyze_repo_health (GitHub metrics), AND
2) call tavily_search for "<owner/repo> security vulnerabilities CVE", AND
3) call tavily_search for "<owner/repo> maintenance status and community sentiment".
A recommendation made without checking for known vulnerabilities is incomplete and
unsafe. Never skip the security searches. Call all required tools now.

Query: {q}"""


async def research_node(state: dict) -> dict:
    async with mcp_tools() as (_session, github_tools):
        tools = github_tools + [tavily_tool]
        tmap = {t.name: t for t in tools}
        llm = research_llm().bind_tools(tools)
        print("[Single-shot] ONE model call — we run whatever tools it returns, then stop (no second turn).")
        msg = await llm.ainvoke(STRONG.format(q=state["user_query"]))
        print(f"[Single-shot] model returned {len(msg.tool_calls)} tool call(s) in ONE shot: {[t['name'] for t in msg.tool_calls]}")

        github, web = [], []
        for tc in msg.tool_calls:
            args = _sanitize_tool_args(tc["name"], tc.get("args", {}))
            if tc["name"] in tmap:
                res = str(await tmap[tc["name"]].ainvoke(args))
                (web if "tavily" in tc["name"] else github).append(res)
        print("[Single-shot] done — the model never saw these results. Straight to the advisory.")
        return {"github_data": str(github), "web_research": str(web)}


if __name__ == "__main__":
    run("VERSION 2 — STRONG PROMPT  (model usually researches security -- but not guaranteed)", research_node)
