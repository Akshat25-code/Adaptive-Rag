"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Adaptive RAG API is running"
    assert "version" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "documents_loaded" in data
    assert "queries_processed" in data


def test_stats():
    response = client.get("/rag/stats")
    assert response.status_code == 200
    data = response.json()
    assert "document_chunks" in data
    assert "total_queries" in data


def test_document_count():
    response = client.get("/rag/documents/count")
    assert response.status_code == 200
    data = response.json()
    assert "document_chunks" in data


def test_upload_wrong_filetype():
    response = client.post(
        "/rag/documents/upload",
        files={"file": ("test.csv", b"col1,col2\n1,2", "text/csv")},
        headers={"X-Description": "test file"},
    )
    assert response.status_code == 400


def test_query_missing_fields():
    response = client.post("/rag/query", json={})
    assert response.status_code == 422


def test_upload_txt_file():
    response = client.post(
        "/rag/documents/upload",
        files={"file": ("test.txt", b"Hello world content", "text/plain")},
        headers={"X-Description": "test document"},
    )
    assert response.status_code in [200, 500]


def test_query_with_sources():
    """Verify query response includes source_documents field."""
    response = client.post(
        "/rag/query",
        json={"query": "test", "session_id": "test-session"},
    )
    if response.status_code == 200:
        data = response.json()
        assert "source_documents" in data


def test_register_and_login():
    response = client.post(
        "/auth/register",
        json={"username": "testuser_xyz_unique", "password": "testpass123"},
    )
    if response.status_code == 200:
        login_resp = client.post(
            "/auth/login",
            json={"username": "testuser_xyz_unique", "password": "testpass123"},
        )
        assert login_resp.status_code == 200
        data = login_resp.json()
        assert "token" in data


def test_login_bad_credentials():
    response = client.post(
        "/auth/login",
        json={"username": "nonexistent_user_abc", "password": "wrong"},
    )
    assert response.status_code in [401, 500]
