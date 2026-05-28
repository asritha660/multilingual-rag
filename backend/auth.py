"""
Authentication module for the Multilingual RAG Assistant.

Handles:
  - password hashing / verification (bcrypt via passlib)
  - JWT token creation and decoding (python-jose)
  - a FastAPI dependency that protects endpoints
"""

import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-env")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # tokens valid for 24 hours

# bcrypt hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Tells FastAPI how clients send the token (Authorization: Bearer <token>)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# --- Password helpers ---
def hash_password(plain_password: str) -> str:
    """Turn a plain password into a secure bcrypt hash for storage."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)


# --- Token helpers ---
def create_access_token(username: str) -> str:
    """Create a signed JWT that encodes the username and an expiry."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    FastAPI dependency. Decodes and verifies the token from the request.
    Returns the username if valid, otherwise raises 401 Unauthorized.
    Use it by adding `user: str = Depends(get_current_user)` to an endpoint.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
