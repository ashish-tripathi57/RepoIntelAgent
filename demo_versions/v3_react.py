"""
VERSION 3 — REAL ReAct AGENT.
GitHub + Tavily tools, in a multi-step loop. The model calls tools, SEES the results,
and decides what to do next — so it can check GitHub first and THEN go research
security based on what it found. Smarter and more autonomous than the single-shot
versions, but still probabilistic: the model could still decide it has "enough."
"""
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from _shared import mcp_tools, research_llm, tavily_tool, _sanitize_tool_args, run

SYSTEM = """You are a due-diligence research analyst. Investigate every repository in
the user's request. For EACH repo: call get_repo_info, analyze_repo_health, and one or
two tavily_search calls for security/CVEs and maintenance.
IMPORTANT: call each tool AT MOST ONCE per repo. As soon as you have that data for every
repo, STOP calling tools and summarize your findings — do NOT keep re-searching the
same things."""

MAX_STEPS = 4  # hard cap — a guardrail against runaway loops


async def research_node(state: dict) -> dict:
    async with mcp_tools() as (_session, github_tools):
        tools = github_tools + [tavily_tool]
        tmap = {t.name: t for t in tools}
        llm = research_llm().bind_tools(tools)

        messages = [SystemMessage(content=SYSTEM), HumanMessage(content=state["user_query"])]
        github, web = [], []
        seen = set()  # GUARDRAIL: remember (tool, args) we've run, so we never repeat one

        # vvv THE REACT LOOP — this is the whole difference from the single-shot versions vvv
        for step in range(MAX_STEPS):           # (1) call the model REPEATEDLY, not once
            print(f"\n{'='*60}\n[ReAct] TURN {step + 1} — sending {len(messages)} messages to the model"
                  f"  (history grows each turn -> THAT is the chaining)\n{'='*60}")
            # GUARDRAIL: model calls can fail (e.g. Gemini 503 "high demand"). Retry once,
            # then degrade gracefully (use the data we have) instead of crashing the demo.
            ai = None
            for attempt in range(2):
                try:
                    ai = await llm.ainvoke(messages)  # it reasons over the FULL history so far
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"[ReAct] model error ({type(e).__name__}); retrying in 4s...")
                        await asyncio.sleep(4)
                    else:
                        print(f"[ReAct] model still erroring ({type(e).__name__}) — proceeding with data gathered so far.")
            if ai is None:
                break
            messages.append(ai)
            if not ai.tool_calls:                # (2) the MODEL decides when it is done
                print(f"[ReAct] model has enough — no more tools. DONE after {step + 1} turn(s).")
                break
            print(f"[ReAct] model chose {len(ai.tool_calls)} tool(s): {[t['name'] for t in ai.tool_calls]}")
            for tc in ai.tool_calls:
                args = _sanitize_tool_args(tc["name"], tc.get("args", {}))
                key = (tc["name"], str(sorted(args.items())))
                if key in seen:                  # GUARDRAIL: refuse to re-run an identical call
                    print(f"   (skip) {tc['name']} {args} — already done; telling the model to move on")
                    messages.append(ToolMessage(content="Already retrieved above. Do NOT call this again; use the data you have and conclude.", tool_call_id=tc["id"]))
                    continue
                seen.add(key)
                tool = tmap.get(tc["name"])
                if tool is None:
                    messages.append(ToolMessage(content="unknown tool", tool_call_id=tc["id"]))
                    continue
                res = str(await tool.ainvoke(args))
                print(f"   -> ran {tc['name']}, feeding {len(res)} chars back to the model")

                # (a) bookkeeping: sort the result into web vs github for the final report
                if "tavily" in tc["name"]:
                    web.append(res)
                else:
                    github.append(res)

                # (3) THE REACT LINE: feed the result back into the conversation so the
                #     model SEES it and reasons on the next turn. A single-shot agent can't.
                messages.append(ToolMessage(content=res[:4000], tool_call_id=tc["id"]))
        else:
            # runs ONLY if the loop finished all MAX_STEPS turns without the model ever
            # saying "done" -> we hit the hard cap. This is the runaway-loop guardrail.
            print(f"[ReAct] reached the MAX_STEPS cap ({MAX_STEPS}) — stopping to prevent a runaway loop.")
        # ^^^ loop exits 3 ways: (a) model stops asking for tools, (b) a model error, (c) MAX_STEPS cap ^^^

        return {"github_data": str(github), "web_research": str(web)}


if __name__ == "__main__":
    run("VERSION 3 — ReAct LOOP  (model reasons across tool results; multi-step)", research_node)
