from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_container, get_db, rate_limited_key
from app.container import Container
from app.core.errors import AppError
from app.db.models import ApiKey, Document
from app.schemas.documents import DocumentOut, SearchHit, SearchRequest, SearchResponse
from app.services.ingestion import extract_text, split_text

router = APIRouter(prefix="/v1", tags=["documents"])


@router.post("/documents", response_model=DocumentOut, status_code=201)
def upload_document(
    file: UploadFile,
    key: ApiKey = Depends(rate_limited_key),
    container: Container = Depends(get_container),
    db: Session = Depends(get_db),
) -> Document:
    settings = container.settings
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise AppError(413, "file_too_large", f"Max upload size is {settings.max_upload_mb} MB.")

    filename = file.filename or "upload"
    text = extract_text(filename, data)
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    vectors = container.embedder.embed_documents(chunks)

    doc = Document(
        owner_id=key.id,
        filename=filename,
        content_type=file.content_type or "",
        n_chunks=len(chunks),
    )
    db.add(doc)
    db.flush()
    container.store.add(db, owner_id=key.id, document_id=doc.id, texts=chunks, vectors=vectors)
    db.commit()
    return doc


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(
    key: ApiKey = Depends(rate_limited_key), db: Session = Depends(get_db)
) -> list[Document]:
    stmt = select(Document).where(Document.owner_id == key.id).order_by(Document.created_at)
    return list(db.scalars(stmt))


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    key: ApiKey = Depends(rate_limited_key),
    db: Session = Depends(get_db),
) -> None:
    doc = db.get(Document, document_id)
    if doc is None or doc.owner_id != key.id:
        raise AppError(404, "not_found", "Document not found.")
    db.delete(doc)
    db.commit()


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    key: ApiKey = Depends(rate_limited_key),
    container: Container = Depends(get_container),
) -> SearchResponse:
    hits = container.retriever.search(key.id, body.query, body.k, body.document_ids)
    return SearchResponse(
        hits=[
            SearchHit(
                document_id=h.document_id,
                filename=h.filename,
                ordinal=h.ordinal,
                score=round(h.score, 4),
                text=h.text,
            )
            for h in hits
        ]
    )
