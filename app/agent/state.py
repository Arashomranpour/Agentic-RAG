from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # persisted conversation
    question: str  # standalone version of the latest user question
    search_query: str  # current retrieval query (changes after a rewrite)
    route: str  # retrieve | web | direct
    documents: list[dict]  # retrieved passages
    web_results: list[dict]  # web search results
    relevant: bool  # did the grader accept the passages?
    rewrites: int
    steps: list[str]  # node trace for this turn
