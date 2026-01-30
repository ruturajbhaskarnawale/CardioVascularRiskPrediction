
from fastapi import APIRouter, HTTPException, status
from ..models.schemas import UserCreate, UserLogin
from ..services.auth_service import auth_service

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserCreate):
    created = auth_service.create_user(user.username, user.password)
    if not created:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User created successfully"}

@router.post("/login")
async def login(user: UserLogin):
    user_data = auth_service.login_user(user.username, user.password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # In a real app, we would return a JWT token here.
    # For now, we just return success to indicate valid credentials.
    return {"message": "Login successful", "username": user.username}
