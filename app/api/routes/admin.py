from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.errors import AppError
from app.core.security import generate_api_key
from app.db.models import ApiKey
from app.schemas.admin import KeyCreate, KeyCreated, KeyOut

router = APIRouter(prefix="/v1/admin/keys", tags=["admin"], dependencies=[Depends(require_admin)])


@router.post("", response_model=KeyCreated, status_code=201)
def create_key(body: KeyCreate, db: Session = Depends(get_db)) -> KeyCreated:
    plain, prefix, key_hash = generate_api_key()
    key = ApiKey(name=body.name, prefix=prefix, key_hash=key_hash)
    db.add(key)
    db.commit()
    return KeyCreated(
        id=key.id,
        name=key.name,
        prefix=key.prefix,
        revoked=key.revoked,
        created_at=key.created_at,
        api_key=plain,
    )


@router.get("", response_model=list[KeyOut])
def list_keys(db: Session = Depends(get_db)) -> list[ApiKey]:
    return list(db.scalars(select(ApiKey).order_by(ApiKey.created_at)))


@router.delete("/{key_id}", status_code=204)
def revoke_key(key_id: str, db: Session = Depends(get_db)) -> None:
    key = db.get(ApiKey, key_id)
    if key is None:
        raise AppError(404, "not_found", "API key not found.")
    key.revoked = True
    db.commit()
