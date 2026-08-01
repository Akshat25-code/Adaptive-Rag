"""Tests for Pydantic models."""

from src.models.grade import Grade
from src.models.query_request import QueryRequest
from src.models.route_identifier import RouteIdentifier
from src.models.verification_result import VerificationResult


def test_grade_model():
    g = Grade(binary_score="yes")
    assert g.binary_score == "yes"

    g2 = Grade(binary_score="no")
    assert g2.binary_score == "no"


def test_route_identifier():
    r = RouteIdentifier(route="index")
    assert r.route == "index"

    r2 = RouteIdentifier(route="general")
    assert r2.route == "general"

    r3 = RouteIdentifier(route="search")
    assert r3.route == "search"


def test_query_request():
    q = QueryRequest(query="What is RAG?", session_id="abc123")
    assert q.query == "What is RAG?"
    assert q.session_id == "abc123"


def test_verification_result():
    v = VerificationResult(faithful=True, explanation="All facts match context.")
    assert v.faithful is True
    assert "facts" in v.explanation

    v2 = VerificationResult(faithful=False, explanation="Missing info.")
    assert v2.faithful is False
