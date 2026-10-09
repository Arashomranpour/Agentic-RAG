import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, rate_limited_key
from app.container import Container
from app.db.models import ApiKey
from app.schemas.chat import ChatRequest, ChatResponse, ThreadResponse
from app.services.chat import (
    build_sources,
    final_answer,
    graph_config,
    history_messages,
    initial_state,
    new_thread_id,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    key: ApiKey = Depends(rate_limited_key),
    container: Container = Depends(get_container),
) -> ChatResponse:
    thread_id = body.thread_id or new_thread_id()
    state = container.graph.invoke(initial_state(body.question), graph_config(key.id, thread_id))
    return ChatResponse(
        thread_id=thread_id,
        answer=final_answer(state),
        standalone_question=state["question"],
        route=state["route"],
        steps=state["steps"],
        sources=build_sources(state),
    )


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    key: ApiKey = Depends(rate_limited_key),
    container: Container = Depends(get_container),
) -> StreamingResponse:
    """Server-Sent Events: `step` per graph node, `token` per generated token,
    then `sources` and `done` (or `error`)."""
    thread_id = body.thread_id or new_thread_id()
    config = graph_config(key.id, thread_id)

    def events() -> Iterator[str]:
        try:
            for mode, payload in container.graph.stream(
                initial_state(body.question), config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node in payload:
                        yield _sse("step", {"node": node})
                else:
                    chunk, meta = payload
                    if meta.get("langgraph_node") == "generate" and isinstance(chunk.content, str):
                        if chunk.content:
                            yield _sse("token", {"text": chunk.content})
            state = container.graph.get_state(config).values
            yield _sse("sources", [s.model_dump() for s in build_sources(state)])
            yield _sse("done", {"thread_id": thread_id, "route": state.get("route", "")})
        except Exception:
            logger.exception("streaming chat failed")
            yield _sse("error", {"message": "The agent failed to produce an answer."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    key: ApiKey = Depends(rate_limited_key),
    container: Container = Depends(get_container),
) -> ThreadResponse:
    snapshot = container.graph.get_state(graph_config(key.id, thread_id))
    return ThreadResponse(thread_id=thread_id, messages=history_messages(snapshot.values or {}))
