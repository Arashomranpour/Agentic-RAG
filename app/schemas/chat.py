from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Reuse to continue a conversation; omit to start a new one.",
    )


class Source(BaseModel):
    type: Literal["document", "web"]
    title: str
    snippet: str
    score: float | None = None
    url: str | None = None
    document_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    standalone_question: str = Field(description="The question after resolving chat history.")
    route: str
    steps: list[str]
    sources: list[Source]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadResponse(BaseModel):
    thread_id: str
    messages: list[Message]
