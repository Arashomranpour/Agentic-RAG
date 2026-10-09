import uuid

from langchain_core.messages import HumanMessage

from app.agent.nodes import message_text
from app.schemas.chat import Message, Source


def new_thread_id() -> str:
    return uuid.uuid4().hex[:16]


def graph_config(owner_id: str, thread_id: str) -> dict:
    # The owner prefix isolates conversations between API keys.
    return {"configurable": {"thread_id": f"{owner_id}:{thread_id}", "owner_id": owner_id}}


def initial_state(question: str) -> dict:
    """Per-turn inputs; resets everything except the persisted message history."""
    return {
        "messages": [HumanMessage(question)],
        "question": question,
        "search_query": question,
        "route": "",
        "documents": [],
        "web_results": [],
        "relevant": False,
        "rewrites": 0,
        "steps": [],
    }


def build_sources(state: dict) -> list[Source]:
    sources: list[Source] = []
    if state.get("relevant"):
        for d in state.get("documents", []):
            sources.append(
                Source(
                    type="document",
                    title=d["filename"],
                    snippet=d["text"][:300],
                    score=d["score"],
                    document_id=d["document_id"],
                )
            )
    for r in state.get("web_results", []):
        sources.append(Source(type="web", title=r["title"], snippet=r["snippet"], url=r["url"]))
    return sources


def final_answer(state: dict) -> str:
    return message_text(state["messages"][-1])


def history_messages(state_values: dict) -> list[Message]:
    out: list[Message] = []
    for m in state_values.get("messages", []):
        if m.type in ("human", "ai") and message_text(m):
            out.append(
                Message(role="user" if m.type == "human" else "assistant", content=message_text(m))
            )
    return out
