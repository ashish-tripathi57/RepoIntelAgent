# The 4 versions — a reliability ladder for "will the agent check security?"

Same agent, same MCP server, same Qwen advisory. The **only** thing that changes is
how the research step decides to do the security check. Run them in order and watch
the security research go from *"maybe"* to *"guaranteed."*

> Your message to the room: **"The more it matters, the further down this ladder you go."**

## Run (from the RepoIntelAgent folder, venv active)
```
python demo_versions/v1_naive.py       # may skip security -> confident, blind GO
python demo_versions/v2_prompted.py    # strong prompt -> usually checks, not guaranteed
python demo_versions/v3_react.py       # real ReAct loop -> reasons across results
python demo_versions/v4_enforced.py    # forced in code -> ALWAYS checks
```

Use the same query for all four so the contrast is clean, e.g.:
```
Should we adopt BerriAI/litellm? Any security or maintenance risks? Compare it to maximhq/bifrost.
```

## What each proves
| Version | Strategy | Outcome |
| --- | --- | --- |
| **1 — Naive** | All 4 tools available, soft prompt, one shot | Model **ignores** the web tool → confident GO with "no security concerns" it never checked |
| **2 — Strong prompt** | + Tavily, firm "you MUST" prompt, one shot | Model *usually* searches — but it's still its choice; run it a few times to see it occasionally skip |
| **3 — ReAct loop** | + multi-step: sees results, decides next | Smarter, more autonomous; checks GitHub then researches based on findings — still probabilistic |
| **4 — Enforced** | + the code forces a security search per repo | The check **cannot** be skipped — guaranteed, every run |

## The talking track
1. **v1:** *"It gave me a confident GO and said there were no security risks — it never looked. That's how AI fails: not a crash, a confident wrong answer."*
2. **v2:** *"So I tell it, firmly, that it must check. Now it usually does… but 'usually' isn't something you bet production on."*
3. **v3:** *"A real agent loop lets it reason across results — much better. Still probabilistic."*
4. **v4:** *"For the thing that actually matters, I stop hoping and enforce it in code. You prompt for the common case; you guarantee the critical one."*

## Notes
- These are teaching copies; your original `agent.py` is untouched and still works.
- v3 (the ReAct loop) is the newest code — test it first tonight; if it misbehaves,
  v1→v2→v4 alone still tell the whole story.
- All four reuse your **local Qwen** advisory agent, so the "research in the cloud,
  judgment on a local model" point lands in every version.
