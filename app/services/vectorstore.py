from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document


@dataclass
class Hit:
    chunk_id: int
    document_id: str
    filename: str
    ordinal: int
    text: str
    score: float


class SqlVectorStore:
    """Brute-force cosine search over embeddings stored in SQL.

    Vectors are L2-normalised, so cosine similarity is a dot product. This is
    plenty for thousands of chunks per key; to scale up, implement the same
    three methods on top of pgvector / Qdrant / FAISS.
    """

    def add(
        self,
        session: Session,
        *,
        owner_id: str,
        document_id: str,
        texts: list[str],
        vectors: np.ndarray,
    ) -> None:
        session.add_all(
            Chunk(
                document_id=document_id,
                owner_id=owner_id,
                ordinal=i,
                text=text,
                embedding=vectors[i].astype(np.float32).tobytes(),
            )
            for i, text in enumerate(texts)
        )

    def count(self, session: Session, owner_id: str) -> int:
        return session.scalar(select(func.count(Chunk.id)).where(Chunk.owner_id == owner_id)) or 0

    def search(
        self,
        session: Session,
        *,
        owner_id: str,
        query_vector: np.ndarray,
        k: int,
        document_ids: list[str] | None = None,
    ) -> list[Hit]:
        stmt = select(Chunk.id, Chunk.embedding).where(Chunk.owner_id == owner_id)
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        rows = session.execute(stmt).all()
        if not rows:
            return []

        ids = [r.id for r in rows]
        matrix = np.vstack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])
        scores = matrix @ query_vector.astype(np.float32)

        top = np.argsort(-scores)[: min(k, len(rows))]
        top_ids = [ids[i] for i in top]
        score_by_id = {ids[i]: float(scores[i]) for i in top}

        chunks = session.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(top_ids))
        ).all()
        hits = [
            Hit(
                chunk_id=c.id,
                document_id=c.document_id,
                filename=filename,
                ordinal=c.ordinal,
                text=c.text,
                score=score_by_id[c.id],
            )
            for c, filename in chunks
        ]
        return sorted(hits, key=lambda h: -h.score)
