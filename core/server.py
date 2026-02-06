import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env files
load_dotenv()

# Import the routers
from core.api.routes import convert_router_v1, convert_router_v2, convert_router_v3, document_types_router
from core.api.routes.api import router as ocr_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(convert_router_v1)
app.include_router(convert_router_v2)
app.include_router(convert_router_v3)
app.include_router(document_types_router)
app.include_router(ocr_router)

## Local Development (run from the root directory)
# PYTHONPATH=. uvicorn core.server:app --host 0.0.0.0 --reload --log-level debug