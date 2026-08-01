"""Tests for auth module."""

from src.api.auth import _create_jwt, _decode_jwt, _hash_password, _verify_password


def test_password_hashing():
    hashed = _hash_password("mypassword")
    assert ":" in hashed
    assert _verify_password("mypassword", hashed) is True
    assert _verify_password("wrongpassword", hashed) is False


def test_different_passwords_different_hashes():
    h1 = _hash_password("pass1")
    h2 = _hash_password("pass2")
    assert h1 != h2


def test_jwt_roundtrip():
    token = _create_jwt("testuser")
    payload = _decode_jwt(token)
    assert payload["sub"] == "testuser"
    assert "exp" in payload
    assert "iat" in payload


def test_jwt_invalid_token():
    try:
        _decode_jwt("invalid.token.here")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_jwt_tampered_signature():
    token = _create_jwt("user1")
    parts = token.split(".")
    parts[2] = "tampered"
    try:
        _decode_jwt(".".join(parts))
        assert False, "Should have raised"
    except ValueError:
        pass
