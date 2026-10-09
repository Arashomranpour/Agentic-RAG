import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.core.config import Settings
from app.core.errors import AppError

_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "what", "how", "who", "for"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


class FakeChatModel(BaseChatModel):
    """Deterministic stand-in for the LLM so tests and demos run offline.

    It recognises each agent step through the [TASK:...] tag in the system prompt.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _respond(self, messages: list[BaseMessage]) -> str:
        system = next((m.content for m in messages if m.type == "system"), "")
        user = next((m.content for m in reversed(messages) if m.type == "human"), "")
        system, user = str(system), str(user)

        if "[TASK:CONDENSE]" in system:
            return user.split("Latest question:")[-1].strip()
        if "[TASK:ROUTE]" in system:
            q = user.split("Question:")[-1].strip().lower()
            if re.match(r"^(hi|hello|hey|thanks|thank you)\b", q):
                return "direct"
            if any(w in q for w in ("latest", "today", "news", "weather", "price of")):
                return "web"
            return "retrieve"
        if "[TASK:GRADE]" in system:
            question, _, passages = user.partition("Passages:")
            return "yes" if _words(question) & _words(passages) else "no"
        if "[TASK:REWRITE]" in system:
            return user.split("Question:")[-1].strip() + " overview"
        if "Context:" in system:
            context = system.split("Context:", 1)[1].strip()
            return f"According to the sources: {context[:160]}"
        if "No documents or web results" in system:
            return "Hello! I could not find anything relevant in your sources."
        return "OK"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._respond(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for word in self._respond(messages).split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            if run_manager:
                run_manager.on_llm_new_token(word + " ", chunk=chunk)
            yield chunk


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_backend == "fake":
        return FakeChatModel()
    if not settings.groq_api_key:
        raise AppError(500, "config_error", "GROQ_API_KEY is not set (or use LLM_BACKEND=fake).")
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.2)
