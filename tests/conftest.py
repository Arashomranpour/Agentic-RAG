import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

ADMIN = "test-admin-token"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        checkpoint_path=str(tmp_path / "checkpoints.db"),
        admin_token=ADMIN,
        llm_backend="fake",
        embedding_backend="fake",
        rate_limit_per_minute=1000,
        max_upload_mb=1,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def make_key(client: TestClient, name: str = "tester") -> dict:
    resp = client.post("/v1/admin/keys", json={"name": name}, headers={"X-Admin-Token": ADMIN})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return {"X-API-Key": body["api_key"]}


@pytest.fixture
def auth(client) -> dict:
    return make_key(client)


def upload(client, headers, name: str, text: str):
    return client.post(
        "/v1/documents", headers=headers, files={"file": (name, text.encode(), "text/plain")}
    )
