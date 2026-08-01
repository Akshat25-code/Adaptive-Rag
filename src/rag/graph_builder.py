"""
Graph builder module for the adaptive RAG system.
"""

import logging

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

from src.models.state import State
from src.rag.nodes import (
    general_llm,
    generate,
    grade,
    query_classifier,
    retriever_node,
    rewrite_query,
    web_search,
)
from src.tools.graph_tools import doc_tool, routing_tool, verify_answer

logger = logging.getLogger(__name__)


graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")

graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_conditional_edges("generate", verify_answer)
graph.add_edge("general_llm", END)

builder = graph.compile()
