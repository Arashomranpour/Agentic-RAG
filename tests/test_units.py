import numpy as np
import pytest

from app.core.errors import AppError
from app.core.security import SlidingWindowLimiter, generate_api_key, hash_api_key
from app.services.embeddings import FakeEmbedder
from app.services.ingestion import extract_text, split_text


def test_api_key_generation_and_hashing():
    plain, prefix, key_hash = generate_api_key()
    assert plain.startswith("rag_") and plain.startswith(prefix)
    assert key_hash == hash_api_key(plain) and plain not in key_hash


def test_rate_limiter_blocks_after_limit():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
    assert limiter.allow("k") and limiter.allow("k")
    assert not limiter.allow("k")
    assert limiter.allow("other")


def test_fake_embedder_similarity_orders_by_overlap():
    emb = FakeEmbedder()
    docs = emb.embed_documents(["cats love fish", "stock markets crashed today"])
    q = emb.embed_query("do cats like fish")
    scores = docs @ q
    assert scores[0] > scores[1]
    assert np.isclose(np.linalg.norm(docs[0]), 1.0)


def test_split_text_respects_chunk_size():
    text = ("Sentence number one. " * 200).strip()
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)


def test_extract_text_rejects_unknown_extension_and_empty():
    with pytest.raises(AppError) as exc:
        extract_text("malware.exe", b"x")
    assert exc.value.status_code == 415
    with pytest.raises(AppError) as exc:
        extract_text("empty.txt", b"   ")
    assert exc.value.status_code == 422
    assert extract_text("a.md", b"# hi") == "# hi"


def test_router_prompt_lists_uploaded_documents():
    """Regression: with documents present the router must be told about them,
    otherwise real models answer 'direct' for questions about the user's files."""
    from langchain_core.messages import AIMessage

    from app.agent.nodes import AgentNodes
    from app.core.config import Settings

    seen: list[str] = []

    class SpyLLM:
        def invoke(self, messages):
            seen.append(messages[0].content)
            return AIMessage(content="retrieve")

    class StubRetriever:
        def document_names(self, owner_id, limit=20):
            return ["atlas.txt", "budget.pdf"]

    nodes = AgentNodes(SpyLLM(), StubRetriever(), None, Settings())
    update = nodes.route(
        {"question": "main risk for Atlas?", "steps": []}, {"configurable": {"owner_id": "o"}}
    )

    assert "atlas.txt, budget.pdf" in seen[0]
    assert update["route"] == "retrieve"


def test_condense_sees_only_user_questions():
    """Regression: assistant replies must not leak into the standalone question."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.nodes import AgentNodes
    from app.core.config import Settings

    seen: list[str] = []

    class SpyLLM:
        def invoke(self, messages):
            seen.append(messages[1].content)
            return AIMessage(content="Who leads Project Atlas?")

    nodes = AgentNodes(SpyLLM(), None, None, Settings())
    update = nodes.condense(
        {
            "messages": [
                HumanMessage("What is the main risk for Atlas?"),
                AIMessage("The risk is the data migration from the legacy wiki."),
                HumanMessage("And who leads that project?"),
            ],
            "steps": [],
        }
    )

    assert "Atlas" in seen[0] and "legacy wiki" not in seen[0]
    assert update["question"] == "Who leads Project Atlas?"
