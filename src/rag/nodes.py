"""
Graph nodes for the adaptive RAG system.
"""

import logging

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.rag.react_agent import get_current_executor
from src.rag.retriever_setup import get_retriever

logger = logging.getLogger(__name__)
config = Config()


def query_classifier(state: State):
    question = state["messages"][-1].content
    retriever = get_retriever()
    context = retriever.invoke(question)
    logger.info("Retrieved context for classification")
    logger.debug("Context: %s", context)

    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(template=config.prompt("classify_prompt"), input_variables=["question", "context"])
    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    logger.info("Query classified as: %s", result.route)

    return {"messages": state["messages"], "route": result.route, "latest_query": question}


def general_llm(state: State):
    result = llm.invoke(state["messages"])
    logger.info("General LLM response generated")
    return {"messages": result}


def retriever_node(state: State):
    query = state["latest_query"]
    agent = get_current_executor()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    output_messages = result.get("messages", [])
    last_message = output_messages[-1] if output_messages else None
    output_text = last_message.content if last_message else ""

    tool_calls = []
    source_documents = []
    for msg in output_messages:
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls"):
            for tc in msg.additional_kwargs["tool_calls"]:
                tool_calls.append(
                    {
                        "tool": tc.get("function", {}).get("name", "unknown"),
                        "input": tc.get("function", {}).get("arguments", ""),
                    }
                )

        if hasattr(msg, "response_metadata") and msg.response_metadata.get("source_documents"):
            for doc in msg.response_metadata["source_documents"]:
                source_documents.append(
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0),
                    }
                )

    new_message = AIMessage(
        content=output_text,
        additional_kwargs={"tool_calls": tool_calls},
    )

    return {
        "messages": [new_message],
        "retrieved_context": output_text,
        "source_documents": source_documents,
    }


def grade(state: State):
    grading_prompt = PromptTemplate(template=config.prompt("grading_prompt"), input_variables=["question", "context"])
    context = state.get("retrieved_context", state["messages"][-1].content)
    question = state["latest_query"]

    llm_with_grade = llm.with_structured_output(Grade)
    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": context})

    logger.info("Grading result: %s", result.binary_score)
    return {"messages": state["messages"], "binary_score": result.binary_score}


def rewrite_query(state: State):
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(template=config.prompt("rewrite_prompt"), input_variables=["query"])
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    logger.info("Query rewritten: %s", result.content[:100])

    return {"latest_query": result.content}


def generate(state: State):
    context = state.get("retrieved_context", state["messages"][-1].content)

    generate_prompt = PromptTemplate(template=config.prompt("generate_prompt"), input_variables=["context"])

    generate_chain = generate_prompt | llm
    result = generate_chain.invoke({"context": context})

    retry_count = (state.get("retry_count", 0) or 0) + 1
    logger.info("Generated answer (attempt %d)", retry_count)
    return {
        "messages": [{"role": "assistant", "content": result.content}],
        "retry_count": retry_count,
        "source_documents": state.get("source_documents", []),
    }


def web_search(state: State):
    search_tool = TavilySearchResults()
    result = search_tool.invoke(state["latest_query"])

    contents = [item["content"] for item in result if "content" in item]
    sources = [{"source": item.get("url", "web"), "page": 0} for item in result if "url" in item]
    joined = "\n\n".join(contents)
    logger.info("Web search returned %d results", len(contents))

    return {
        "messages": [{"role": "assistant", "content": joined}],
        "retrieved_context": joined,
        "source_documents": sources,
    }
