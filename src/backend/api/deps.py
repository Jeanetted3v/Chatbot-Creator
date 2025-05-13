from fastapi import Request, HTTPException
from fastapi import WebSocket
import logging
from hydra import initialize, compose
from src.backend.chat.service_container import ServiceContainer

logger = logging.getLogger(__name__)


def get_config():
    """Dependency to provide Hydra configuration."""
    with initialize(version_base=None, config_path="./../../../config"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True)
    return cfg
    

def get_service_container(request: Request) -> ServiceContainer:
    if not getattr(request.app.state, "startup_complete", False):
        raise HTTPException(
            status_code=503, 
            detail="Service is starting up. Please try again in a moment."
        )
    if not hasattr(request.app.state, "service_container"):
        raise HTTPException(
            status_code=500,
            detail="Service container not available"
        )
    
    return request.app.state.service_container


async def get_websocket_service_container(websocket: WebSocket) -> ServiceContainer:
    """Dependency to get the service container from app state for WebSocket connections."""
    if not hasattr(websocket.app.state, "service_container"):
        logger.warning("Service container not available for WebSocket - app may still be initializing")
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return None
    return websocket.app.state.service_container


