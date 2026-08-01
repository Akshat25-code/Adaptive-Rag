"""Tests for graph routing logic."""

import pytest
from langchain_core.messages import AIMessage

from src.tools.graph_tools import doc_tool, routing_tool, verify_answer


def _make_state(**overrides):
    """Create minimal state dict for testing."""
    base = {
        "messages": [],
        "binary_score": None,
        "route": None,
        "latest_query": None,
        "retry_count": 0,
        "retrieved_context": None,
        "source_documents": None,
    }
    base.update(overrides)
    return base


def test_routing_tool_index():
    state = _make_state(route="index")
    assert routing_tool(state) == "retriever"


def test_routing_tool_general():
    state = _make_state(route="general")
    assert routing_tool(state) == "general_llm"


def test_routing_tool_search():
    state = _make_state(route="search")
    assert routing_tool(state) == "web_search"


def test_routing_tool_unknown_falls_to_search():
    state = _make_state(route="something_else")
    assert routing_tool(state) == "web_search"


def test_doc_tool_yes():
    state = _make_state(binary_score="yes")
    assert doc_tool(state) == "generate"


def test_doc_tool_no():
    state = _make_state(binary_score="no")
    assert doc_tool(state) == "rewrite"


def test_verify_answer_general_route():
    state = _make_state(route="general", messages=[AIMessage(content="test answer")])
    assert verify_answer(state) == "__end__"


@pytest.mark.skip(reason="Requires valid OpenAI API key")
def test_verify_answer_faithful():
    state = _make_state(
        route="index",
        retry_count=0,
        retrieved_context="relevant context",
        messages=[AIMessage(content="answer based on context")],
    )
    result = verify_answer(state)
    assert result in ["__end__", "generate"]


def test_verify_answer_max_retries():
    state = _make_state(
        route="index",
        retry_count=2,
        retrieved_context="some context",
        messages=[AIMessage(content="some answer")],
    )
    assert verify_answer(state) == "__end__"


def test_state_with_source_documents():
    state = _make_state(
        source_documents=[{"source": "doc.pdf", "page": 1}],
        route="index",
    )
    assert len(state["source_documents"]) == 1
    assert state["source_documents"][0]["source"] == "doc.pdf"
