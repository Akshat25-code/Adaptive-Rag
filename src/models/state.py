"""
State model for the graph-based RAG system.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict):
    """State schema for the RAG graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    binary_score: str | None
    route: str | None
    latest_query: str | None
    retry_count: int | None
    retrieved_context: str | None
    source_documents: list[dict] | None
