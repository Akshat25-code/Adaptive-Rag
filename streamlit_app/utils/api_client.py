"""
API client for communicating with backend services.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def create_user(username: str, password: str) -> bool:
    """Register new user."""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/register",
            json={"username": username, "password": password},
        )
        if response.status_code == 200:
            return True
        logger.error("Register failed: %s - %s", response.status_code, response.text)
        return False
    except requests.RequestException as e:
        logger.exception("Register request failed: %s", e)
        return False


def login_user(username: str, password: str) -> dict:
    """Login and get JWT token."""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": username, "password": password},
        )
        if response.status_code == 200:
            return response.json()
        logger.error("Login failed: %s", response.status_code)
        return None
    except requests.RequestException as e:
        logger.exception("Login request failed: %s", e)
        return None


def query_backend(query: str, session_id: str) -> dict:
    """
    Send query to RAG backend.

    Returns dict with 'content', 'route', 'time_seconds', 'sources'.
    """
    try:
        response = requests.post(
            f"{BASE_URL}/rag/query",
            json={"query": query, "session_id": session_id},
            allow_redirects=False,
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "content": data["result"]["content"],
                "route": data.get("route", "unknown"),
                "time_seconds": data.get("time_seconds", 0),
                "sources": data.get("source_documents", []),
            }
        error_content = f"Error: {response.status_code} - {response.text}"
        return {"content": error_content, "route": "error", "time_seconds": 0, "sources": []}
    except requests.RequestException as e:
        logger.exception("Query request failed: %s", e)
        error_content = f"Connection error: {e}"
        return {"content": error_content, "route": "error", "time_seconds": 0, "sources": []}


def document_upload_rag(file, description: str) -> bool:
    """Upload document to RAG system."""
    headers = {"X-Description": description}
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{BASE_URL}/rag/documents/upload", files=files, headers=headers)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.exception("Upload failed: %s", e)
        return False
