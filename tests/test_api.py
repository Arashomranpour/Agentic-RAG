import json

from tests.conftest import make_key, upload

DOC = (
    "Project Atlas launches in March. The Atlas budget is 2 million dollars.\n\n"
    "The Atlas team is led by Dana Reyes and has twelve engineers."
)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_auth_is_required(client):
    assert client.get("/v1/documents").status_code == 401
    assert client.get("/v1/documents", headers={"X-API-Key": "rag_nope"}).status_code == 401


def test_admin_requires_token(client):
    assert client.post("/v1/admin/keys", json={"name": "x"}).status_code == 401
    ok = client.post(
        "/v1/admin/keys", json={"name": "x"}, headers={"X-Admin-Token": "test-admin-token"}
    )
    assert ok.status_code == 201 and ok.json()["api_key"].startswith("rag_")


def test_revoked_key_stops_working(client):
    headers = {"X-Admin-Token": "test-admin-token"}
    created = client.post("/v1/admin/keys", json={"name": "temp"}, headers=headers).json()
    key = {"X-API-Key": created["api_key"]}
    assert client.get("/v1/documents", headers=key).status_code == 200
    assert client.delete(f"/v1/admin/keys/{created['id']}", headers=headers).status_code == 204
    assert client.get("/v1/documents", headers=key).status_code == 401


def test_upload_list_search_delete(client, auth):
    resp = upload(client, auth, "atlas.txt", DOC)
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["filename"] == "atlas.txt" and doc["n_chunks"] >= 1

    assert [d["id"] for d in client.get("/v1/documents", headers=auth).json()] == [doc["id"]]

    hits = client.post("/v1/search", json={"query": "Atlas budget", "k": 2}, headers=auth).json()[
        "hits"
    ]
    assert hits and "budget" in hits[0]["text"].lower()

    assert client.delete(f"/v1/documents/{doc['id']}", headers=auth).status_code == 204
    assert client.post("/v1/search", json={"query": "Atlas"}, headers=auth).json()["hits"] == []


def test_upload_validation(client, auth):
    bad_type = client.post(
        "/v1/documents", headers=auth, files={"file": ("x.exe", b"MZ", "application/octet-stream")}
    )
    assert bad_type.status_code == 415
    too_big = client.post(
        "/v1/documents",
        headers=auth,
        files={"file": ("big.txt", b"a" * (1024 * 1024 + 10), "text/plain")},
    )
    assert too_big.status_code == 413
    assert upload(client, auth, "empty.txt", "   ").status_code == 422


def test_chat_retrieves_from_documents(client, auth):
    upload(client, auth, "atlas.txt", DOC)
    body = client.post(
        "/v1/chat", json={"question": "What is the Atlas budget?"}, headers=auth
    ).json()
    assert body["route"] == "retrieve"
    assert "grade:yes" in body["steps"] and body["steps"][-1] == "generate"
    assert body["sources"] and body["sources"][0]["type"] == "document"
    assert body["sources"][0]["title"] == "atlas.txt"
    assert "According to the sources" in body["answer"]


def test_chat_direct_for_small_talk(client, auth):
    body = client.post("/v1/chat", json={"question": "hello there"}, headers=auth).json()
    assert body["route"] == "direct" and body["sources"] == []


def test_chat_without_documents_falls_back_to_direct(client, auth):
    body = client.post(
        "/v1/chat", json={"question": "What is the Atlas budget?"}, headers=auth
    ).json()
    assert body["route"] == "direct" and body["sources"] == []


def test_irrelevant_question_triggers_rewrite_then_gives_up(client, auth):
    upload(client, auth, "atlas.txt", DOC)
    body = client.post(
        "/v1/chat", json={"question": "zebra quantum xylophone"}, headers=auth
    ).json()
    assert "rewrite" in body["steps"]
    assert body["sources"] == []  # irrelevant passages are never cited


def test_thread_memory_persists_and_is_listed(client, auth):
    first = client.post("/v1/chat", json={"question": "hello"}, headers=auth).json()
    tid = first["thread_id"]
    client.post("/v1/chat", json={"question": "hi again", "thread_id": tid}, headers=auth)
    msgs = client.get(f"/v1/threads/{tid}", headers=auth).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "hello"


def test_data_is_isolated_between_keys(client):
    alice, bob = make_key(client, "alice"), make_key(client, "bob")
    doc = upload(client, alice, "atlas.txt", DOC).json()

    assert client.get("/v1/documents", headers=bob).json() == []
    assert client.post("/v1/search", json={"query": "Atlas"}, headers=bob).json()["hits"] == []
    assert client.delete(f"/v1/documents/{doc['id']}", headers=bob).status_code == 404

    tid = client.post("/v1/chat", json={"question": "hello"}, headers=alice).json()["thread_id"]
    assert client.get(f"/v1/threads/{tid}", headers=bob).json()["messages"] == []


def test_stream_emits_steps_tokens_sources_done(client, auth):
    upload(client, auth, "atlas.txt", DOC)
    with client.stream(
        "POST", "/v1/chat/stream", json={"question": "Atlas budget?"}, headers=auth
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert "step" in events and "token" in events
    assert events[-2:] == ["sources", "done"]
    done = json.loads(text.split("event: done\ndata: ")[1].split("\n")[0])
    assert done["route"] == "retrieve" and done["thread_id"]


def test_rate_limit(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings.rate_limit_per_minute = 3
    with TestClient(create_app(settings)) as c:
        headers = make_key(c)
        codes = [c.get("/v1/documents", headers=headers).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_request_id_header_is_set(client):
    assert (
        client.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"]
        == "abc123"
    )
