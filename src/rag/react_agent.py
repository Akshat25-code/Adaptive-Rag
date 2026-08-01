"""
ReAct agent setup for document retrieval and question answering.
"""

from langchain.agents import create_agent

from src.config.settings import Config
from src.llms.openai import llm
from src.rag.retriever_setup import get_retriever

config = Config()


def get_current_executor():
    tools = [get_retriever()]
    system_prompt = config.prompt("system_prompt")
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
