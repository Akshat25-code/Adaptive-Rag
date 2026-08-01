"""
Tools for graph routing and document grading.
"""

import logging
from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.llms.openai import llm
from src.models.state import State
from src.models.verification_result import VerificationResult

logger = logging.getLogger(__name__)
config = Config()


def routing_tool(state: State) -> Literal["retriever", "general_llm", "web_search"]:
    """Route graph to appropriate node based on query classification."""
    if state["route"] == "index":
        return "retriever"
    elif state["route"] == "general":
        return "general_llm"
    else:
        return "web_search"


def doc_tool(state: State) -> Literal["rewrite", "generate"]:
    """Determine whether query needs rewriting based on grading score."""
    score = state["binary_score"]
    logger.info("Routing based on grade score: %s", score)
    if score == "yes":
        return "generate"
    else:
        return "rewrite"


def verify_answer(state: State) -> Literal["__end__", "generate"]:
    """
    Verify whether final answer is faithful to retrieved context.
    Max 2 retries to prevent infinite loops.
    """
    if state["route"] == "general":
        return "__end__"

    retry_count = state.get("retry_count", 0) or 0
    if retry_count >= 2:
        logger.warning("Max retries reached, ending verification loop")
        return "__end__"

    question = state["latest_query"]
    context = state.get("retrieved_context", "")
    final_answer = state["messages"][-1].content

    verify_prompt = PromptTemplate(
        template=config.prompt("verify_prompt"), input_variables=["question", "context", "final_answer"]
    )
    llm_with_verification = llm.with_structured_output(VerificationResult)
    verify_chain = verify_prompt | llm_with_verification

    result = verify_chain.invoke({"question": question, "context": context, "final_answer": final_answer})

    if result.faithful:
        logger.info("Answer verified as faithful")
        return "__end__"
    else:
        logger.info("Answer not faithful, regenerating (attempt %d)", retry_count + 1)
        return "generate"
