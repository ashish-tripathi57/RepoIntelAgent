import pytest
from unittest.mock import patch, MagicMock
from concurrent.futures import TimeoutError


@patch('agents.advisory_agent._invoke_with_timeout')
@patch('agents.advisory_agent.ChatOllama')
def test_advisory_agent_formats_report(mock_ollama, mock_invoke_timeout):
    from agents.advisory_agent import advisory_node

    mock_response = MagicMock(content="GO: Good project")
    mock_invoke_timeout.return_value = mock_response

    dummy_state = {
        "user_query": "Check langchain security issues",
        "github_data": "{'issues': 5, 'health': 'good'}",
        "web_research": "Recent bug found.",
        "final_report": ""
    }

    result = advisory_node(dummy_state)
    assert result["final_report"] == "GO: Good project"
    mock_invoke_timeout.assert_called_once()


@patch('agents.advisory_agent._invoke_with_timeout')
@patch('agents.advisory_agent.ChatGoogleGenerativeAI')
@patch('agents.advisory_agent.ChatOllama')
@patch.dict('os.environ', {'ADVISORY_PRIMARY_MODEL': 'qwen3:14b', 'ADVISORY_FALLBACK_MODEL': 'gemini-2.5-flash'})
def test_advisory_agent_failover_to_fallback(mock_ollama, mock_chat, mock_invoke_timeout):
    """Primary Ollama model fails, fallback Gemini model succeeds."""
    from agents.advisory_agent import advisory_node

    fallback_response = MagicMock(content="CONDITIONAL: Needs review")
    mock_invoke_timeout.side_effect = [TimeoutError("primary timed out"), fallback_response]

    dummy_state = {
        "user_query": "test query",
        "github_data": "some data",
        "web_research": "some research",
        "final_report": ""
    }

    result = advisory_node(dummy_state)

    assert result["final_report"] == "CONDITIONAL: Needs review"
    mock_ollama.assert_called_once()
    mock_chat.assert_called_once()


@patch('agents.advisory_agent._invoke_with_timeout')
@patch('agents.advisory_agent.ChatGoogleGenerativeAI')
@patch('agents.advisory_agent.ChatOllama')
@patch.dict('os.environ', {'ADVISORY_PRIMARY_MODEL': 'qwen3:14b', 'ADVISORY_FALLBACK_MODEL': 'gemini-2.5-flash'})
def test_advisory_agent_both_models_fail_graceful_exit(mock_ollama, mock_chat, mock_invoke_timeout):
    """Both Ollama and Gemini fail — returns error message."""
    from agents.advisory_agent import advisory_node

    mock_invoke_timeout.side_effect = TimeoutError("timed out")

    dummy_state = {
        "user_query": "test query",
        "github_data": "some data",
        "web_research": "some research",
        "final_report": ""
    }

    result = advisory_node(dummy_state)

    assert "Both primary and fallback models failed" in result["final_report"]
    mock_ollama.assert_called_once()
    mock_chat.assert_called_once()
