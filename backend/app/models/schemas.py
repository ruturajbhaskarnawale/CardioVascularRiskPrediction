
from pydantic import BaseModel
from typing import Optional, List, Union

# Auth Schemas
class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str

# Prediction Schemas
class PredictionInput(BaseModel):
    full_name: Optional[str] = "Anonymous"
    phone_number: Optional[str] = ""
    age: float
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    gender: str  # "Male" or "Female"
    cholesterol: Union[int, str] # 1, 2, 3 or mapped string
    gluc: Union[int, str]        # 1, 2, 3 or mapped string
    smoke: Union[int, str]       # 0, 1 or mapped string
    alco: Union[int, str]        # 0, 1 or mapped string
    active: Union[int, str]      # 0, 1 or mapped string
    stress: Optional[Union[int, str]] = 1 # Added for completeness based on Main_App

class PredictionResponse(BaseModel):
    result: str
    probability: float
    risk_level: str
    risk_color: str
    recommendations: str
    factors: str
    screening: str
    resources: str

class BulkPredictionResponse(BaseModel):
    total_records: int
    avg_risk: float
    predictions: List[dict] # Simplified for now

# Resource Schemas
class ResourceQuery(BaseModel):
    city_name: str
    resource_type: str
    max_distance_km: int = 10

class ResourceItem(BaseModel):
    Facility_Name: str
    Facility_Type: str
    Facility_Address: Optional[str]
    distance_km: float
    Latitude: float
    Longitude: float
