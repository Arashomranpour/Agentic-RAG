from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agent import prompts
from app.agent.state import AgentState
from app.agent.tools import WebSearch
from app.core.config import Settings
from app.services.retrieval import Retriever


def message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


class AgentNodes:
    """Graph nodes with their dependencies injected, so each one is easy to test."""

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Retriever,
        web_search: WebSearch | None,
        settings: Settings,
    ):
        self.llm = llm
        self.retriever = retriever
        self.web_search = web_search
        self.settings = settings

    # ---- helpers -------------------------------------------------------
    def _ask(self, system: str, user: str) -> str:
        reply = self.llm.invoke([SystemMessage(system), HumanMessage(user)])
        return message_text(reply).strip()

    @staticmethod
    def _owner(config: RunnableConfig) -> str:
        return config["configurable"]["owner_id"]

    @staticmethod
    def _trace(state: AgentState, name: str) -> list[str]:
        return [*state.get("steps", []), name]

    # ---- nodes ---------------------------------------------------------
    def condense(self, state: AgentState) -> dict:
        messages = state["messages"]
        latest = message_text(messages[-1])
        if len(messages) <= 1:
            return {
                "question": latest,
                "search_query": latest,
                "steps": self._trace(state, "condense"),
            }
        # Only the user's own earlier questions: assistant replies carry extra details that
        # leak into the rewritten question and make retrieval too narrow.
        earlier = [message_text(m)[:500] for m in messages[:-1] if m.type == "human"]
        history = "\n".join(f"- {q}" for q in earlier[-self.settings.history_messages :])
        standalone = self._ask(
            prompts.CONDENSE, f"Earlier user questions:\n{history}\n\nLatest question: {latest}"
        )
        standalone = standalone or latest
        return {
            "question": standalone,
            "search_query": standalone,
            "steps": self._trace(state, "condense"),
        }

    def route(self, state: AgentState, config: RunnableConfig) -> dict:
        names = self.retriever.document_names(self._owner(config))
        has_docs = bool(names)
        system = prompts.ROUTE.format(documents=", ".join(names) if names else "(none)")
        decision = self._ask(system, f"Question: {state['question']}").lower()
        route = next((r for r in ("retrieve", "web", "direct") if r in decision), "retrieve")

        # Never promise a tool that cannot work.
        if route == "web" and self.web_search is None:
            route = "retrieve" if has_docs else "direct"
        if route == "retrieve" and not has_docs:
            route = "web" if self.web_search else "direct"
        return {"route": route, "steps": self._trace(state, f"route:{route}")}

    def retrieve(self, state: AgentState, config: RunnableConfig) -> dict:
        hits = self.retriever.search(
            self._owner(config), state["search_query"], self.settings.top_k
        )
        documents = [
            {
                "text": h.text,
                "filename": h.filename,
                "document_id": h.document_id,
                "ordinal": h.ordinal,
                "score": round(h.score, 4),
            }
            for h in hits
        ]
        return {"documents": documents, "steps": self._trace(state, "retrieve")}

    def grade(self, state: AgentState) -> dict:
        documents = state.get("documents", [])
        if not documents:
            return {"relevant": False, "steps": self._trace(state, "grade:no")}
        passages = "\n\n".join(f"- {d['text'][:600]}" for d in documents)
        verdict = self._ask(
            prompts.GRADE, f"Question: {state['question']}\n\nPassages:\n{passages}"
        ).lower()
        relevant = verdict.startswith("yes")
        return {
            "relevant": relevant,
            "steps": self._trace(state, f"grade:{'yes' if relevant else 'no'}"),
        }

    def rewrite(self, state: AgentState) -> dict:
        query = self._ask(prompts.REWRITE, f"Question: {state['question']}") or state["question"]
        return {
            "search_query": query,
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": self._trace(state, "rewrite"),
        }

    def web(self, state: AgentState) -> dict:
        results = self.web_search(state["question"]) if self.web_search else []
        return {"web_results": results, "steps": self._trace(state, "web_search")}

    def generate(self, state: AgentState) -> dict:
        blocks: list[str] = []
        if state.get("relevant"):
            for d in state.get("documents", []):
                blocks.append(f"[{len(blocks) + 1}] ({d['filename']}) {d['text']}")
        for r in state.get("web_results", []):
            blocks.append(f"[{len(blocks) + 1}] ({r['title']}) {r['snippet']} <{r['url']}>")

        if blocks:
            instructions = prompts.WITH_CONTEXT.format(context="\n\n".join(blocks))
        else:
            instructions = prompts.NO_CONTEXT
        system = SystemMessage(prompts.ANSWER.format(context_instructions=instructions))

        history = state["messages"][-self.settings.history_messages :]
        reply = self.llm.invoke([system, *history])
        return {"messages": [reply], "steps": self._trace(state, "generate")}

    # ---- routing functions --------------------------------------------
    def after_route(self, state: AgentState) -> str:
        return {"retrieve": "retrieve", "web": "web", "direct": "generate"}[state["route"]]

    def after_grade(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("rewrites", 0) < self.settings.max_rewrites:
            return "rewrite"
        return "web" if self.web_search else "generate"
