
from fastapi import APIRouter, HTTPException, Query
from typing import List
from ..models.schemas import ResourceItem
from ..services.resource_service import resource_service

router = APIRouter(
    prefix="/resources",
    tags=["Resources"]
)

@router.get("/", response_model=List[ResourceItem])
async def find_resources(
    city: str = Query(..., description="City name (e.g. Mumbai, Pune)"),
    resource_type: str = Query("Hospital", description="Type of resource"),
    radius: int = Query(10, description="Search radius in km")
):
    results = resource_service.find_resources(city, resource_type, radius)
    return results
