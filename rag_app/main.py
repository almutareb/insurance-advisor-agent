from fastapi import FastAPI
from rag_app.api.api import api_router as api_router_v1
from fastapi.responses import HTMLResponse
from rag_app.utils.config import settings
from rag_app.templates.chat import chat_html
#from app.core.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)
#BACKEND_CORS_ORIGINS = ["*"]
# CORS Middleware setup for allowing frontend requests
# ToDO: replace with settings.BACKEND_CORS_ORIGINS once core/config.py is implemented
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/", tags=["Root"])
async def root():
    """
    Simple root endpoint to verify the API is running.
    """
    return {"message": "API is running"}

@app.get("/chat", response_class=HTMLResponse)
async def chat():

    return chat_html

# Include the versioned API router from api.py
app.include_router(api_router_v1)