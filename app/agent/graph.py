from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentNodes
from app.agent.state import AgentState
from app.agent.tools import WebSearch
from app.core.config import Settings
from app.services.retrieval import Retriever


def build_graph(
    *,
    llm: BaseChatModel,
    retriever: Retriever,
    web_search: WebSearch | None,
    checkpointer: BaseCheckpointSaver,
    settings: Settings,
):
    """Corrective RAG:

    condense -> route -+-> retrieve -> grade -+-> generate
                       |                      +-> rewrite -> retrieve
                       +-> web ---------------+-> web (after failed retries)
                       +-> generate (direct answer)
    """
    nodes = AgentNodes(llm, retriever, web_search, settings)
    builder = StateGraph(AgentState)

    builder.add_node("condense", nodes.condense)
    builder.add_node("route", nodes.route)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("rewrite", nodes.rewrite)
    builder.add_node("web", nodes.web)
    builder.add_node("generate", nodes.generate)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "route")
    builder.add_conditional_edges(
        "route", nodes.after_route, {"retrieve": "retrieve", "web": "web", "generate": "generate"}
    )
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", nodes.after_grade, {"generate": "generate", "rewrite": "rewrite", "web": "web"}
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("web", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)
