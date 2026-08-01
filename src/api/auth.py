"""
Authentication routes — JWT-based auth replacing external Rust service.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.db.mongo_client import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRY_HOURS = 24

users_collection = db["users"]


class AuthRequest(BaseModel):
    username: str
    password: str


def _hash_password(password: str) -> str:
    """Hash password with SHA-256 + salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify password against stored hash."""
    salt, hashed = stored.split(":")
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == hashed


def _create_jwt(username: str) -> str:
    """Create JWT token."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    now = datetime.now(UTC)
    payload_data = {
        "sub": username,
        "exp": int((now + timedelta(hours=JWT_EXPIRY_HOURS)).timestamp()),
        "iat": int(now.timestamp()),
    }
    payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip("=")
    signature = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{signature}"


def _decode_jwt(token: str) -> dict:
    """Decode and verify JWT token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header, payload, signature = parts

        expected_sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
        if signature != expected_sig:
            raise ValueError("Invalid signature")

        padding = 4 - len(payload) % 4
        payload_data = json.loads(base64.urlsafe_b64decode(payload + "=" * padding))

        if payload_data["exp"] < int(datetime.now(UTC).timestamp()):
            raise ValueError("Token expired")

        return payload_data
    except Exception as e:
        raise ValueError(f"Token validation failed: {e}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency: extract and verify user from JWT."""
    try:
        payload = _decode_jwt(credentials.credentials)
        return payload["sub"]
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/register")
async def register(req: AuthRequest):
    """Register new user."""
    existing = await users_collection.find_one({"username": req.username})
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    await users_collection.insert_one(
        {
            "username": req.username,
            "password": _hash_password(req.password),
            "created_at": datetime.now(UTC),
        }
    )

    logger.info("User registered: %s", req.username)
    return {"message": "User created successfully"}


@router.post("/login")
async def login(req: AuthRequest):
    """Login and receive JWT token."""
    user = await users_collection.find_one({"username": req.username})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not _verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_jwt(req.username)
    logger.info("User logged in: %s", req.username)
    return {"token": token, "username": req.username}
