"""
OMNIVIS — Authentication Module
JWT + OAuth2 (Google/GitHub) + Role-Based Access Control
"""
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.session import get_db
from db.models import User, UserRole
from api.schemas import TokenPayload

# ── Config ─────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "omnivis-super-secret-key-change-in-production-!@#$%")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# OAuth2 config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# ── Password Hashing ──────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    return secrets.token_urlsafe(48)


# ── Token Creation ─────────────────────────────────────────────
def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Dependencies ───────────────────────────────────────────────
async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Authenticate via JWT token or API key."""

    if api_key:
        key_hash = hash_api_key(api_key)
        result = await db.execute(
            select(User).where(User.api_key_hash == key_hash, User.is_active == True)
        )
        user = result.scalar_one_or_none()
        if user:
            return user

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    result = await db.execute(
        select(User).where(User.id == payload.sub, User.is_active == True)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Optional authentication — returns None if not authenticated."""
    try:
        return await get_current_user(token, api_key, db)
    except HTTPException:
        return None


def require_role(*roles: UserRole):
    """Dependency factory for role-based access control."""
    async def role_checker(user: User = Depends(get_current_user)):
        if user.role not in [r.value for r in roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' not authorized. Required: {[r.value for r in roles]}"
            )
        return user
    return role_checker


# ── OAuth2 Helpers ─────────────────────────────────────────────
async def get_google_user_info(code: str) -> dict:
    """Exchange Google OAuth code for user info."""
    import httpx
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/google/callback"),
                "grant_type": "authorization_code",
            }
        )
        token_data = token_resp.json()

        # Get user info
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"}
        )
        return user_resp.json()


async def get_github_user_info(code: str) -> dict:
    """Exchange GitHub OAuth code for user info."""
    import httpx
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "code": code,
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
            },
            headers={"Accept": "application/json"}
        )
        token_data = token_resp.json()

        # Get user info
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token_data['access_token']}"}
        )
        user_info = user_resp.json()

        # Get email (may be private)
        email_resp = await client.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"Bearer {token_data['access_token']}"}
        )
        emails = email_resp.json()
        primary_email = next((e["email"] for e in emails if e["primary"]), None)
        user_info["email"] = primary_email or user_info.get("email")

        return user_info
