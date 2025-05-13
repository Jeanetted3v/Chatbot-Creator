import logging
from datetime import datetime
from fastapi import APIRouter
from fastapi import APIRouter, HTTPException, Depends, Query
from src.backend.models.human_agent import AgentType, MessageRole, ChatTurn, ToggleReason
from src.backend.chat.service_container import ServiceContainer
from src.backend.models.human_agent import ToggleReason, AgentType
from src.backend.api.serialization import serialize_mongodb_doc
from src.backend.api.deps import get_service_container


logger = logging.getLogger(__name__)
router = APIRouter()


async def human_takeover(
    session_id: str,
    reason: ToggleReason,
    services: ServiceContainer = None
) -> str:
    """Handle the transfer of a conversation to a human agent.
    This function can be called from either customer or staff routers.
    
    Args:
        session_id: The session identifier
        reason: The reason for the handover (enum ToggleReason)
        services: The service container instance
        
    Returns:
        str: The takeover message that was sent to the customer
    """
    logger.info(f"[Human Takeover] reason: {reason}")
    session = services.active_sessions.get(session_id)
    # Skip if already in human mode
    if session.current_agent == AgentType.HUMAN:
        logger.info(f"Session {session_id} already handled by human agent")
        return "Session already handled by human agent"
    
    # Use the human handler to perform the transfer
    transfer_result = await services.human_handler.transfer_to_human(
        session_id,
        reason,
    )
    
    if transfer_result is False:
        logger.error(f"Failed to transfer session {session_id} to human agent")
        return "Failed to transfer to human agent"
    
    logger.info(f"Session {session_id} transferred to human agent. Reason: {reason}")
    # Get the standard message (your HumanAgentHandler already adds this to chat history)
    takeover_message = "Chat transferred to human agent"
    return takeover_message


@router.get("/chat/history")
async def get_chat_history(
    session_id: str,
    customer_id: str,
    limit: int = Query(20, ge=1, le=100),
    services: ServiceContainer = Depends(get_service_container)
):
    """Get conversation history for a session (staff view)"""
    
    try:
        chat_history = await services.get_chat_history(session_id, customer_id)
        history = await chat_history.get_recent_turns(limit)

        logger.info(f"Retrieved {len(history)} raw history items for session {session_id}")
        # First, serialize any ObjectId in the raw history data
        serialized_history = [serialize_mongodb_doc(turn) for turn in history]
        
        # Convert to API format
        result = []
        for turn in serialized_history:
            chat_turn = ChatTurn(
                role=turn.get("role", MessageRole.SYSTEM),
                content=turn.get("content", ""),
                timestamp=turn.get("timestamp", datetime.now()),
                customer_id=customer_id,
                session_id=session_id,
                metadata=turn.get("metadata")
            )
            result.append(chat_turn)
        logger.info(f"Converted {len(result)} items to API format for session {session_id}")
        logger.info(f"Chat history for session {session_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting chat history for staff: {e}")
        raise HTTPException(status_code=500, detail=str(e))