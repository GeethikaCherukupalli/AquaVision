import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router

# This is the "app" variable Uvicorn is looking for
app = FastAPI(title="AquaVision AI Maritime Pipeline", version="2.0")

# Enable CORS so the React frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the directory for saving overlay images exists and is accessible
os.makedirs("static/overlays", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Connect our API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "online", "message": "AquaVision API is running."}