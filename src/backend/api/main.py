"""To test run: python -m src.backend.api.main
Interact via SwaggerUi: http://localhost:8000/chat/docs
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.backend.utils.logging import setup_logging
from src.backend.api.deps import get_config
from src.backend.api import customer_router, staff_router, utils_router
from src.backend.chat.service_container import ServiceContainer
from src.backend.api import websocket_router

setup_logging()
logger = logging.getLogger(__name__)
cfg = get_config()
ORIGINS = ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles initialization and cleanup of service container.
    """
    app.state.startup_complete = False
    try:
        logger.info("Creating service container")
        service_container = ServiceContainer(cfg)
        await service_container.initialize()
    
        # Store in app state
        app.state.service_container = service_container
        app.state.startup_complete = True
        logger.info("Service container initialized and ready")
        print("==== ABOUT TO YIELD ====")  # This should show before the yield
        yield
        print("==== AFTER YIELD ====") 
        # Shutdown code
        app.state.startup_complete = False 
        if hasattr(app.state, "service_container"):
            await app.state.service_container.cleanup()
            logger.info("Service container cleaned up")
    except Exception as e:
        logger.error(f"Error during application initialization: {e}", exc_info=True)
        raise

app = FastAPI(
    title="Agentic Edu Chatbot",
    description="API an agentic chatbot for educational company.",
    version="1.0",
    docs_url="/chat/docs",
    openapi_url="/chat/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to ORIGINS in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(customer_router.router, prefix="/customer", tags=["customer"])
app.include_router(staff_router.router, prefix="/staff", tags=["staff"])
app.include_router(utils_router.router, prefix="/utils", tags=["utils"])
app.include_router(websocket_router.router, prefix="/ws", tags=["websocket"])


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint for the FastAPI server."""
    return {
        "message": "Welcome to the Agentic Edu Chatbot API",
        "version": 1.0,
        "docs": "chat/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for the FastAPI server."""
    return {"status": "healthy"}


def main() -> None:
    """Main function to run the FastAPI server."""
    uvicorn.run(
        "src.backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=cfg.api.reload,
    )


if __name__ == "__main__":
    main()

