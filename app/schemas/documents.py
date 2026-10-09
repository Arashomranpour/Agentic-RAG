from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    n_chunks: int
    created_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    document_ids: list[str] | None = None


class SearchHit(BaseModel):
    document_id: str
    filename: str
    ordinal: int
    score: float
    text: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]
