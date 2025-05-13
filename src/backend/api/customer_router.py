import logging
from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends

from src.backend.models.api import MessageRequest, MessageResponse
from src.backend.chat.service_container import ServiceContainer
from src.backend.models.human_agent import AgentType, ToggleReason, MessageRole
from src.backend.models.api import SessionResponse
from src.backend.api.deps import get_service_container
from src.backend.api.utils_router import human_takeover

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/session/new", response_model=SessionResponse)
async def create_new_session(
    services: ServiceContainer = Depends(get_service_container)
):
    """Create a new session and return its details"""
    session_data = await services.create_new_session()
    return SessionResponse(**session_data)


@router.post("/message", response_model=MessageResponse)
async def customer_send_message(
    message_request: MessageRequest,
    services: ServiceContainer = Depends(get_service_container)
):
    """Main endpoint, customer send a message and get a response
    
    Routes message to handle_query funciton.
    Being called whenevr a customer sends a message in the chat interface.
    """
    try:
        customer_id = message_request.customer_id
        session_id = message_request.session_id
        # source = message_request.source
        message = message_request.message

        if not session_id:
            session_id = await services.check_session(customer_id)
        
        # Get or create session with the determined session_id
        session = await services.get_or_create_session(session_id, customer_id)
        
        # Process the message using your existing query handler
        response = await services.query_handler.handle_query(
            message,
            session_id,
            customer_id
        )
        # Check if the session was transferred to human during handle_query
        # handle_query set session.current_agent = AgentType.HUMAN
        # and call human_handler.transfer_to_human internally
        
        # If the session was transferred to human but not fully processed,
        # handle the takeover to ensure proper notification and logging
        if (session.current_agent == AgentType.HUMAN and 
            "human agent" in response.lower() and 
            "transferred" in response.lower()):
            
            # Use the shared takeover function
            await human_takeover(
                session_id=session_id,
                reason=ToggleReason.SENTIMENT_BASED,  # This could also be CUSTOMER_REQUEST depending on your logic
                services=services
            )
        
        # Determine the response role based on current agent
        response_role = (
            MessageRole.HUMAN_AGENT 
            if session.current_agent == AgentType.HUMAN 
            else MessageRole.BOT
        )
        
        return MessageResponse(
            message=response,
            session_id=session_id,
            customer_id=customer_id,
            role=response_role,
            current_agent=session.current_agent,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


