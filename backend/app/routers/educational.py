
from fastapi import APIRouter
from ..services.educational_service import get_educational_content

router = APIRouter(
    prefix="/educational",
    tags=["Educational"]
)

@router.get("/")
async def get_content():
    return get_educational_content()
