# routers/staff_router.py
import logging
from typing import List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from src.backend.chat.service_container import ServiceContainer
from src.backend.models.human_agent import AgentType, MessageRole, ToggleReason
from src.backend.models.api import SessionResponse, StaffMessageRequest, MessageResponse, TransferRequest, TakeoverRequest

from src.backend.api.deps import get_service_container
from src.backend.api.utils_router import human_takeover


# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sessions/active", response_model=List[SessionResponse])
async def get_active_sessions(
    services: ServiceContainer = Depends(get_service_container)
):
    """Get all active chat sessions for staff to view"""
    try:
        result = []
        for session in services.active_sessions.values():
            session_response = SessionResponse(
                session_id=session.session_id,
                customer_id=session.customer_id,
                current_agent=session.current_agent,
                start_time=session.start_time,
                last_interaction=session.last_interaction,
                message_count=getattr(session, 'message_count', 0) 
            )
            result.append(session_response)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/message", response_model=MessageResponse)
async def staff_send_message(
    message_request: StaffMessageRequest,
    services: ServiceContainer = Depends(get_service_container)
):
    """Send a message from the human agent to the customer"""
    try:
        session_id = message_request.session_id
        customer_id = message_request.customer_id
        message = message_request.message
        
        # Get session
        session = await services.get_or_create_session(session_id, customer_id)
        chat_history = await services.get_chat_history(session_id, customer_id)
        
        # Ensure session is in human agent mode
        if session.current_agent != AgentType.HUMAN:
            await human_takeover(
                session_id=session_id,
                reason=ToggleReason.AGENT_INITIATED,
                services=services
            )
        
        # Add message to chat history
        await chat_history.add_turn(MessageRole.HUMAN_AGENT, message)
        
        # Update session last interaction time
        session.last_interaction = datetime.now()
        
        # If integrated with WhatsApp, send message via WhatsApp API
        # This would depend on your WhatsApp integration
        # await send_whatsapp_message(customer_id, message)
        
        return MessageResponse(
            message=message,
            session_id=session_id,
            customer_id=customer_id,
            role=MessageRole.HUMAN_AGENT,
            current_agent=AgentType.HUMAN,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error sending human agent message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transfer/bot", response_model=MessageResponse)
async def transfer_to_bot(
    transfer_request: TransferRequest,
    services: ServiceContainer = Depends(get_service_container)
):
    """Transfer conversation from human agent back to bot"""
    logger.info(f"[Router Transfer Bot]Transfer request: {transfer_request}")
    try:
        session_id = transfer_request.session_id
        customer_id = transfer_request.customer_id
        custom_message = transfer_request.message
        
        # Get session and chat history
        session = await services.get_or_create_session(session_id, customer_id)
        chat_history = await services.get_chat_history(session_id, customer_id)
        
        # Only proceed if currently in human mode
        if session.current_agent != AgentType.HUMAN:
            return MessageResponse(
                message="Session already handled by bot",
                session_id=session_id,
                customer_id=customer_id,
                role=MessageRole.SYSTEM,
                current_agent=AgentType.BOT,
                timestamp=datetime.now()
            )
        
        # Transfer to bot using your handler
        transfer_message = await services.human_handler.transfer_to_bot(
            session_id,
            chat_history
        )
        # Update session agent type
        session.current_agent = AgentType.BOT
        
        return MessageResponse(
            message=transfer_message,
            session_id=session_id,
            customer_id=customer_id,
            role=MessageRole.SYSTEM,
            current_agent=AgentType.BOT,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error transferring to bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/takeover", response_model=MessageResponse)
async def take_over_session(
    takeover_request: TakeoverRequest,
    services: ServiceContainer = Depends(get_service_container)
) -> MessageResponse:
    """Staff manually takes over a session that's currently handled by bot

    If staff find that the bot is unable to handle the conversation, they can
    take over the session to provide human assistance
    """
    logger.info(f"[Router Takeover]Takeover request: {takeover_request}")
    try:
        session_id = takeover_request.session_id
        customer_id = takeover_request.customer_id
        
        # Use the shared takeover function
        takeover_message = await human_takeover(
            session_id=session_id,
            reason=ToggleReason.AGENT_INITIATED,
            services=services
        )
        
        # Get the updated session
        session = await services.get_or_create_session(session_id, customer_id)
        
        return MessageResponse(
            message=takeover_message,
            session_id=session_id,
            customer_id=customer_id,
            role=MessageRole.SYSTEM,
            current_agent=session.current_agent,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error taking over session: {e}")
        raise HTTPException(status_code=500, detail=str(e))