from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Document
from app.services.embeddings import Embedder
from app.services.vectorstore import Hit, SqlVectorStore


class Retriever:
    """Embeds a query and searches one owner's chunks. Used by the API and the agent."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        store: SqlVectorStore,
    ):
        self._session_factory = session_factory
        self._embedder = embedder
        self._store = store

    def search(
        self, owner_id: str, query: str, k: int, document_ids: list[str] | None = None
    ) -> list[Hit]:
        vector = self._embedder.embed_query(query)
        with self._session_factory() as session:
            return self._store.search(
                session, owner_id=owner_id, query_vector=vector, k=k, document_ids=document_ids
            )

    def has_documents(self, owner_id: str) -> bool:
        with self._session_factory() as session:
            return self._store.count(session, owner_id) > 0

    def document_names(self, owner_id: str, limit: int = 20) -> list[str]:
        with self._session_factory() as session:
            stmt = select(Document.filename).where(Document.owner_id == owner_id).limit(limit)
            return list(session.scalars(stmt))
