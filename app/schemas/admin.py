from datetime import datetime

from pydantic import BaseModel, Field


class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class KeyOut(BaseModel):
    id: str
    name: str
    prefix: str
    revoked: bool
    created_at: datetime


class KeyCreated(KeyOut):
    api_key: str = Field(description="Shown once. Store it safely - only a hash is kept.")
