from collections.abc import Iterator

from fastapi import Depends, Header, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.container import Container
from app.core.errors import AppError
from app.core.security import admin_token_matches, hash_api_key
from app.db.models import ApiKey


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_db(container: Container = Depends(get_container)) -> Iterator[Session]:
    with container.session_factory() as session:
        yield session


def current_key(
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> ApiKey:
    if not x_api_key:
        raise AppError(401, "missing_api_key", "Send your key in the X-API-Key header.")
    key = db.scalar(select(ApiKey).where(ApiKey.key_hash == hash_api_key(x_api_key)))
    if key is None or key.revoked:
        raise AppError(401, "invalid_api_key", "Invalid or revoked API key.")
    return key


def rate_limited_key(
    key: ApiKey = Depends(current_key),
    container: Container = Depends(get_container),
) -> ApiKey:
    if not container.limiter.allow(key.id):
        raise AppError(429, "rate_limited", "Too many requests. Slow down and retry shortly.")
    return key


def require_admin(
    x_admin_token: str | None = Header(default=None),
    container: Container = Depends(get_container),
) -> None:
    if not container.settings.admin_token:
        raise AppError(404, "admin_disabled", "Admin endpoints are disabled (ADMIN_TOKEN not set).")
    if not admin_token_matches(container.settings.admin_token, x_admin_token):
        raise AppError(401, "invalid_admin_token", "Invalid admin token.")
