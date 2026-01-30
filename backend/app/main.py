
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# Import routers (will be created next)
from .routers import auth, predict, resources, educational

app = FastAPI(
    title="CardioVascular Risk Prediction API",
    description="Backend API for CardioHealth Risk Predictor Pro",
    version="1.0.0"
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(resources.router)
app.include_router(educational.router)

# CORS Configuration
origins = [
    "http://localhost:3000", # Next.js frontend
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure assets dir exists before mounting to avoid error
if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/")
async def root():
    return {"message": "Welcome to CardioHealth Risk Prediction API"}
