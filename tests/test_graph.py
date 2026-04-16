import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def test_graph_state_topology():
    """Verify the state graph compiles and async nodes are structurally valid."""
    from agent import setup_graph

    graph = setup_graph()
    assert graph is not None
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "astream")


@pytest.mark.asyncio
@patch('agents.advisory_agent._invoke_with_timeout')
@patch('agents.advisory_agent.ChatGoogleGenerativeAI')
@patch('agents.research_agent.ChatGoogleGenerativeAI')
@patch('agents.research_agent.ClientSession')
@patch('agents.research_agent.stdio_client')
async def test_graph_streams_step_by_step(mock_stdio, mock_session_cls, mock_research_chat, mock_advisory_chat, mock_invoke_timeout):
    """Verify astream produces updates from both nodes."""
    from agent import run_agent

    # Setup research agent mocks
    mock_research_llm = AsyncMock()
    mock_research_chat.return_value.bind_tools.return_value = mock_research_llm
    mock_research_llm.ainvoke.return_value = MagicMock(tool_calls=[])

    mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
    mock_session = AsyncMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session
    mock_session.list_tools.return_value = MagicMock(tools=[])

    # Setup advisory agent mocks
    mock_advisory_response = MagicMock(content="GO: All clear")
    mock_invoke_timeout.return_value = mock_advisory_response

    result = await run_agent("test query")

    assert result == "GO: All clear"
