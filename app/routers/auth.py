from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import ROLES, Participant, User
from ..security import audit, current_user, decode_token, hash_password, make_token, require_roles, verify_password

router = APIRouter(prefix="/api", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


class RefreshIn(BaseModel):
    refresh_token: str


def user_view(db: Session, u: User) -> dict:
    participant = db.scalars(select(Participant).where(Participant.user_id == u.id)).first()
    return {
        "id": u.id,
        "username": u.username,
        "role": u.role,
        "active": u.active,
        "participant_id": participant.id if participant else None,
    }


@router.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.scalars(select(User).where(User.username == body.username.strip())).first()
    if u is None or not u.active or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Username or password is incorrect.")
    audit(db, u, "login", "user", u.id)
    db.commit()
    return {"access_token": make_token(u), "refresh_token": make_token(u, "refresh"), "user": user_view(db, u)}


@router.post("/auth/refresh")
def refresh(body: RefreshIn, db: Session = Depends(get_db)):
    payload = decode_token(body.refresh_token, "refresh")
    u = db.get(User, int(payload["sub"]))
    if u is None or not u.active:
        raise HTTPException(401, "This account is not active.")
    return {"access_token": make_token(u), "refresh_token": make_token(u, "refresh"), "user": user_view(db, u)}


@router.get("/auth/me")
def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return user_view(db, user)


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8)
    role: str


class UserPatch(BaseModel):
    password: str | None = Field(default=None, min_length=8)
    role: str | None = None
    active: bool | None = None


@router.get("/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(require_roles("admin"))):
    return [user_view(db, u) for u in db.scalars(select(User).order_by(User.id))]


@router.post("/users", status_code=201)
def create_user(body: UserIn, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    if body.role not in ROLES:
        raise HTTPException(422, f"Role must be one of {', '.join(ROLES)}.")
    if db.scalars(select(User).where(User.username == body.username)).first():
        raise HTTPException(409, "That username is taken.")
    u = User(username=body.username, password_hash=hash_password(body.password), role=body.role)
    db.add(u)
    db.flush()
    audit(db, admin, "create", "user", u.id, role=body.role)
    db.commit()
    return user_view(db, u)


@router.patch("/users/{user_id}")
def patch_user(user_id: int, body: UserPatch, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "User not found.")
    if body.role is not None:
        if body.role not in ROLES:
            raise HTTPException(422, f"Role must be one of {', '.join(ROLES)}.")
        u.role = body.role
    if body.active is not None:
        u.active = body.active
    if body.password:
        u.password_hash = hash_password(body.password)
    audit(db, admin, "update", "user", u.id, fields=[k for k, v in body.model_dump().items() if v is not None and k != "password"])
    db.commit()
    return user_view(db, u)
