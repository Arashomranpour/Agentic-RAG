import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.graph import build_graph
from app.agent.tools import WebSearch, make_serper_search
from app.core.config import Settings
from app.core.security import SlidingWindowLimiter
from app.db.session import init_db, make_engine, make_session_factory
from app.services.embeddings import Embedder, build_embedder
from app.services.llm import build_llm
from app.services.retrieval import Retriever
from app.services.vectorstore import SqlVectorStore


@dataclass
class Container:
    """Holds every long-lived dependency. Built once at startup, overridable in tests."""

    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    embedder: Embedder
    store: SqlVectorStore
    retriever: Retriever
    graph: object
    limiter: SlidingWindowLimiter
    checkpoint_conn: sqlite3.Connection

    def close(self) -> None:
        self.checkpoint_conn.close()
        self.engine.dispose()


def build_container(settings: Settings) -> Container:
    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)

    embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
    store = SqlVectorStore()
    retriever = Retriever(session_factory, embedder, store)

    web_search: WebSearch | None = (
        make_serper_search(settings.serper_api_key) if settings.serper_api_key else None
    )

    Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    graph = build_graph(
        llm=build_llm(settings),
        retriever=retriever,
        web_search=web_search,
        checkpointer=SqliteSaver(checkpoint_conn),
        settings=settings,
    )

    return Container(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        embedder=embedder,
        store=store,
        retriever=retriever,
        graph=graph,
        limiter=SlidingWindowLimiter(settings.rate_limit_per_minute),
        checkpoint_conn=checkpoint_conn,
    )
