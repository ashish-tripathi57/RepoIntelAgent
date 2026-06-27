"""
VERSION 4 — ENFORCED GUARDRAIL.
Strong prompt + GitHub + Tavily, AND the code ALWAYS runs a security search for every
repo afterward — no matter what the model did. The critical check can never be
silently skipped, because it's enforced in code, not left to the model's discretion.
This is the production answer: prompt for the common case, ENFORCE the thing that matters.
"""
from _shared import mcp_tools, research_llm, tavily_tool, tavily_search, _sanitize_tool_args, run
from v2_prompted import STRONG


async def research_node(state: dict) -> dict:
    async with mcp_tools() as (_session, github_tools):
        tools = github_tools + [tavily_tool]
        tmap = {t.name: t for t in tools}
        llm = research_llm().bind_tools(tools)
        msg = await llm.ainvoke(STRONG.format(q=state["user_query"]))

        github, web, repos = [], [], set()
        for tc in msg.tool_calls:
            args = _sanitize_tool_args(tc["name"], tc.get("args", {}))
            print(f"[Research] calling {tc['name']} {args}")
            if tc["name"] in tmap:
                res = str(await tmap[tc["name"]].ainvoke(args))
                (web if "tavily" in tc["name"] else github).append(res)
            if args.get("owner") and args.get("repo"):
                repos.add((args["owner"], args["repo"]))

        # --- ENFORCED: always research security for every repo, no matter what ---
        for owner, repo in sorted(repos):
            print(f"[Guardrail] Forcing security web search for {owner}/{repo}")
            web.append(tavily_search(f"{owner}/{repo} security vulnerabilities CVE advisories maintenance status"))

        return {"github_data": str(github), "web_research": str(web)}


if __name__ == "__main__":
    run("VERSION 4 — ENFORCED GUARDRAIL  (security search forced in code -> guaranteed)", research_node)
