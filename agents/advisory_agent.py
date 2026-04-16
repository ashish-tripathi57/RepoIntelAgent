import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def advisory_node(state: dict) -> dict:
    print("\n[Advisory Agent] Synthesizing final report...")

    primary_model = os.environ.get("ADVISORY_PRIMARY_MODEL", "qwen3:14b")
    fallback_model = os.environ.get("ADVISORY_FALLBACK_MODEL", "gemini-2.5-flash")

    prompt = f"""
    Analyze the collected data and provide a business recommendation.
    Outputs MUST include:
    - GO / NO-GO / CONDITIONAL recommendation
    - Risk factors identified from GitHub metrics
    - External concerns found via web search
    - Mitigation suggestions for each risk
    - Overall confidence score (High/Medium/Low)

    Data:
    GitHub Data: {state['github_data']}
    Web Research: {state['web_research']}
    """

    # Failover chain: primary (Ollama local) → fallback (Gemini cloud) → graceful exit
    try:
        print(f"[Advisory Agent] Calling {primary_model} (local/Ollama)...")
        llm = ChatOllama(model=primary_model)
        response = _invoke_with_timeout(llm, prompt, timeout=300)
        print(f"[Advisory Agent] {primary_model} succeeded.")
    except (TimeoutError, Exception) as e:
        print(f"[Advisory Agent] Primary model failed ({type(e).__name__}). Falling back to {fallback_model}...")
        try:
            llm = ChatGoogleGenerativeAI(model=fallback_model)
            response = _invoke_with_timeout(llm, prompt, timeout=300)
            print(f"[Advisory Agent] {fallback_model} succeeded.")
        except (TimeoutError, Exception) as e2:
            print(f"[Advisory Agent] Both models failed. Primary: {type(e).__name__}, Fallback: {type(e2).__name__}")
            return {"final_report": "Error: Both primary and fallback models failed. Please retry later."}

    print("[Advisory Agent] Final recommendation complete.")
    return {"final_report": response.content}


def _invoke_with_timeout(llm, prompt: str, timeout: int):
    """Run a synchronous llm.invoke() with a timeout. Uses shutdown(wait=False) to avoid blocking on slow API responses."""
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(llm.invoke, prompt)
    try:
        result = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
