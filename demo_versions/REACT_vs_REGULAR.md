# ReAct loop vs. single-shot — the real code difference

**It's a structural change, not just the prompt.** The system prompt is different too,
but the essence is the control flow. Here's the contrast, stripped to the core — good
to put side by side on a slide, or to show live in the two files.

## Single-shot (v1, v2, v4): the model writes a to-do list once
```python
msg = await llm.ainvoke(prompt)      # ONE model call
for tc in msg.tool_calls:            # just EXECUTE the batch it asked for
    run(tc)                          # the model NEVER sees the results
# stop
```
That `for` loop is **not** an agent loop — it only runs the list of tool calls the
model produced in a single turn. The model decides everything up front, blind.

## ReAct (v3): the model is IN the loop
```python
messages = [system, user]
for step in range(MAX_STEPS):                      # (1) call the model REPEATEDLY
    ai = await llm.ainvoke(messages)               #     over the growing history
    messages.append(ai)
    if not ai.tool_calls:                          # (2) the MODEL decides when to stop
        break
    for tc in ai.tool_calls:
        result = run(tc)
        messages.append(ToolMessage(result, tc.id))  # (3) FEED RESULTS BACK
    # loop -> the model now reasons about what it just learned
```

## The four differences to point at on screen
| # | Single-shot | ReAct |
| --- | --- | --- |
| 1 | one `llm.ainvoke(...)` | `llm.ainvoke(...)` **inside a loop** |
| 2 | a string prompt | a **growing `messages` history** |
| 3 | tool results discarded | results appended as **`ToolMessage`** and seen by the model |
| 4 | stops after one pass | stops **when the model decides** (no more tool calls), capped by `MAX_STEPS` |

## The line that says it all
> *"In the single-shot version, the model writes a to-do list and we run it — blind.
> In ReAct, the model runs a tool, sees the result, and decides the next move. The
> model is inside the loop. **That loop is what makes it an agent.**"*

This ties straight to your concept slide — **agent = LLM + tools + a loop.** v1/v2/v4
are "LLM + tools." v3 adds the loop.

> Worth saying out loud for credibility: LangGraph's `create_agent` / prebuilt ReAct
> agent does *exactly* this loop for you — v3 is what's under that abstraction. And
> `MAX_STEPS` is your guardrail against the "runaway loop" failure mode from the talk.
