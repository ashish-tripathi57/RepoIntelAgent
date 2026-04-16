import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.asyncio
@patch('agents.research_agent.ChatGoogleGenerativeAI')
@patch('agents.research_agent.ClientSession')
@patch('agents.research_agent.stdio_client')
@patch('agents.research_agent.TavilyClient')
@patch.dict('os.environ', {'TAVILY_API_KEY': 'dummy'})
async def test_research_agent_returns_structured_data(mock_tavily, mock_stdio, mock_session_cls, mock_chat):
    from agents.research_agent import research_node

    mock_llm = AsyncMock()
    mock_chat.return_value.bind_tools.return_value = mock_llm

    mock_llm.ainvoke.return_value = MagicMock(
        tool_calls=[
            {"name": "get_repo_info", "args": {"owner": "foo", "repo": "bar"}},
            {"name": "list_open_issues", "args": {"owner": "foo", "repo": "bar"}},
            {"name": "tavily_search", "args": {"query": "foo"}}
        ]
    )

    mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())

    mock_session = AsyncMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    mock_session.list_tools.return_value = MagicMock(tools=[MagicMock(name="get_repo_info")])
    mock_session.call_tool.return_value = MagicMock(content="Mocked GitHub")

    mock_tavily_inst = MagicMock()
    mock_tavily.return_value = mock_tavily_inst
    mock_tavily_inst.search.return_value = "Mocked Web"

    dummy_state = {
        "user_query": "Check langchain security issues",
        "github_data": "",
        "web_research": "",
        "final_report": ""
    }

    result = await research_node(dummy_state)

    assert "Mocked GitHub" in result["github_data"]
    assert "Mocked Web" in result["web_research"]


@pytest.mark.asyncio
@patch('agents.research_agent.ChatGoogleGenerativeAI')
@patch('agents.research_agent.ClientSession')
@patch('agents.research_agent.stdio_client')
@patch.dict('os.environ', {'RESEARCH_PRIMARY_MODEL': 'gemini-3.1-pro-preview', 'RESEARCH_FALLBACK_MODEL': 'gemini-2.5-flash'})
async def test_research_agent_failover_to_fallback(mock_stdio, mock_session_cls, mock_chat):
    """Primary Gemini model fails, fallback Gemini model succeeds."""
    from agents.research_agent import research_node

    primary_llm = AsyncMock()
    fallback_llm = AsyncMock()

    primary_llm.ainvoke.side_effect = asyncio.TimeoutError()
    fallback_llm.ainvoke.return_value = MagicMock(tool_calls=[])

    call_count = 0
    def make_chat(model):
        nonlocal call_count
        call_count += 1
        mock_bound = MagicMock()
        if call_count == 1:
            mock_bound.bind_tools.return_value = primary_llm
        else:
            mock_bound.bind_tools.return_value = fallback_llm
        return mock_bound

    mock_chat.side_effect = make_chat

    mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
    mock_session = AsyncMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    mock_session.list_tools.return_value = MagicMock(tools=[])

    dummy_state = {
        "user_query": "test query",
        "github_data": "",
        "web_research": "",
        "final_report": ""
    }

    result = await research_node(dummy_state)

    assert "github_data" in result
    assert "web_research" in result
    assert call_count == 2


@pytest.mark.asyncio
@patch('agents.research_agent.ChatGoogleGenerativeAI')
@patch('agents.research_agent.ClientSession')
@patch('agents.research_agent.stdio_client')
@patch.dict('os.environ', {'RESEARCH_PRIMARY_MODEL': 'gemini-3.1-pro-preview', 'RESEARCH_FALLBACK_MODEL': 'gemini-2.5-flash'})
async def test_research_agent_both_models_fail_graceful_exit(mock_stdio, mock_session_cls, mock_chat):
    """Both Gemini models fail — returns whatever was collected."""
    from agents.research_agent import research_node

    failing_llm = AsyncMock()
    failing_llm.ainvoke.side_effect = asyncio.TimeoutError()

    mock_chat.return_value.bind_tools.return_value = failing_llm

    mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
    mock_session = AsyncMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    mock_session.list_tools.return_value = MagicMock(tools=[])

    dummy_state = {
        "user_query": "test query",
        "github_data": "",
        "web_research": "",
        "final_report": ""
    }

    result = await research_node(dummy_state)

    assert result["github_data"] == "[]"
    assert result["web_research"] == "[]"


def test_sanitize_tool_args_fixes_hallucinated_repo():
    from agents.research_agent import _sanitize_tool_args

    args = _sanitize_tool_args("analyze_repo_health", {"owner": "tensorflow", "arg": "tensorflow"})
    assert args == {"owner": "tensorflow", "repo": "tensorflow"}


def test_sanitize_tool_args_passes_correct_args():
    from agents.research_agent import _sanitize_tool_args

    args = _sanitize_tool_args("get_repo_info", {"owner": "foo", "repo": "bar"})
    assert args == {"owner": "foo", "repo": "bar"}


@pytest.mark.asyncio
async def test_missing_tavily_key():
    from agents.research_agent import tavily_tool
    with patch.dict('os.environ', {}, clear=True):
        res = await tavily_tool.ainvoke({"query": "foo"})
        assert "Tavily key missing" in res
