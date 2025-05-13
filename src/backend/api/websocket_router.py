# src/backend/api/websocket_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime
import json
from starlette.websockets import WebSocketState
from src.backend.api.deps import get_websocket_service_container
from src.backend.chat.service_container import ServiceContainer
from src.backend.api.websocket_manager import manager, ConnectionManager
from src.backend.api.utils_router import human_takeover
from src.backend.models.human_agent import ToggleReason
from src.backend.models.api import MessageRole, AgentType
import logging

logger = logging.getLogger(__name__)

router = APIRouter()



async def handle_customer_message(
    services: ServiceContainer,
    session_id: str,
    customer_id: str,
    content: str,
    manager: ConnectionManager,
    message_time: datetime = None
) -> None:
    """Process a message from a customer, broadcast responses to all clients.
    
    1. Processes the customer's message using the QueryHandler
    2. Retrieves or creates the appropriate session
    3. Determines the response role based on current agent type
    4. Broadcasts the original customer message to all clients in the session
    5. Broadcasts the response (from bot or human agent) to all clients in
        the session
    
    Args:
        services: Container with service dependencies for processing messages.
        session_id: Unique identifier for the chat session.
        customer_id: Unique identifier for the customer.
        content: The text content of the customer's message.
        manager: Connection manager for broadcasting messages to WebSocket
            clients.
    
    Note:
        The broadcast_to_session function forwards messages to all clients 
        (both customer and staff interfaces) connected to the same chat session.
    """
    if message_time is None:
        message_time = datetime.now()

    response = await services.query_handler.handle_query(
        content,
        session_id,
        customer_id
    )
    session = await services.get_or_create_session(session_id, customer_id)
    response_role = (
        MessageRole.HUMAN_AGENT 
        if session.current_agent == AgentType.HUMAN 
        else MessageRole.BOT
    )
    # Broadcast the customer's message to all connections
    await manager.broadcast_to_session(
        session_id,
        {
            "type": "new_message",
            "message": {
                "role": MessageRole.USER,
                "content": content,
                "timestamp": message_time.isoformat(),
                "session_id": session_id,
                "customer_id": customer_id
            }
        }
    )
    # Broadcast the response, from bot or human agent, to all connections
    await manager.broadcast_to_session(
        session_id,
        {
            "type": "new_message",
            "message": {
                "role": response_role,
                "content": response,
                "timestamp": message_time.isoformat(),
                "session_id": session_id,
                "customer_id": customer_id
            }
        }
    )


async def handle_staff_message(
    services: ServiceContainer,
    session_id: str,
    customer_id: str,
    content: str,
    manager: ConnectionManager,
    client_message_id: str = None,
    message_time: datetime = None
) -> None:
    """Process a message from a staff member, broadcast it to all clients.
    
    1. Retrieves or creates the appropriate session
    2. Gets the chat history for the session
    3. Ensures the session is in human agent mode (triggers takeover if needed)
    4. Adds the staff message to the chat history
    5. Updates the last interaction timestamp
    6. Broadcasts the staff message to all clients in the session
    
    Args:
        services: Container with service dependencies for processing messages.
        session_id: Unique identifier for the chat session.
        customer_id: Unique identifier for the customer being assisted.
        content: The text content of the staff member's message.
        manager: Connection manager class for broadcasting messages to
            WebSocketclients.
        client_message_id: Optional unique identifier for the message sent by
            the staff member. This can be used for tracking or referencing
            specific messages in the chat history.
    
    Note:
        If the session is currently in bot mode, this function will
        automatically trigger a human takeover before sending the message.
    """
    session = await services.get_or_create_session(session_id, customer_id)
    chat_history = await services.get_chat_history(session_id, customer_id)

    if message_time is None:
        message_time = datetime.now()

    # Ensure session is in human agent mode
    if session.current_agent != AgentType.HUMAN:
        await human_takeover(
            session_id=session_id,
            reason=ToggleReason.AGENT_INITIATED,
            services=services
        )
    
    await chat_history.add_turn(
        MessageRole.HUMAN_AGENT,
        content,
        timestamp=message_time
    )
    session.last_interaction = message_time.isoformat()
    
    # Broadcast the staff message to all connections
    message_data = {
        "type": "new_message",
        "message": {
            "role": MessageRole.HUMAN_AGENT,
            "content": content,
            "timestamp": message_time.isoformat(),
            "session_id": session_id,
            "customer_id": customer_id
        }
    }
    
    if client_message_id:
        message_data["client_message_id"] = client_message_id
    await manager.broadcast_to_session(session_id, message_data)


async def handle_command(
    services: ServiceContainer,
    session_id: str,
    customer_id: str,
    action: str,
    manager: ConnectionManager,
    message_time: datetime = None
) -> None:
    """Process administrative commands from staff members via WebSocket.
    
    Handles various session control commands including agent takeover and 
    transfer operations. Validates command applicability, executes the 
    appropriate action, and sends status notifications to clients.
    
    Supported commands:
        1. "takeover" - Transfers control from bot to human agent
        2. "transfer_to_bot" - Returns control from human agent to bot
    
    Workflow:
        1. Retrieves or creates the session and chat history
        2. Validates if the requested action is applicable to the current state
        3. Executes the requested action if valid
        4. Sends command result notification to staff interface
        5. Broadcasts system messages to all session clients when appropriate
    
    Args:
        services: Container with service dependencies for processing commands.
        session_id: Unique identifier for the chat session.
        customer_id: Unique identifier for the customer in the session.
        action: The command action to perform
            (e.g., "takeover", "transfer_to_bot").
        manager: Connection manager class for broadcasting messages to
            WebSocket clients.
    
    Raises:
        Exception: Catches and logs any errors during command processing,
                  sending error notifications to the staff interface.
    """
    try:
        session = await services.get_or_create_session(session_id, customer_id)
        chat_history = await services.get_chat_history(session_id, customer_id)
        
        if action == "takeover":
            # Only proceed if not already in human mode
            if session.current_agent == AgentType.HUMAN:
                await manager.send_command_message(
                    {
                        "type": "command_result",
                        "action": "takeover",
                        "success": False,
                        "message": "Session already handled by human agent"
                    },
                    session_id
                )
                return
            
            # Use the existing takeover function
            takeover_message = await human_takeover(
                session_id=session_id,
                reason=ToggleReason.AGENT_INITIATED,
                services=services
            )
            
            # Notify the client that the takeover was successful
            await manager.send_command_message(
                {
                    "type": "command_result",
                    "action": "takeover",
                    "success": True,
                    "message": takeover_message
                },
                session_id
            )
            
        elif action == "transfer_to_bot":
            # Only proceed if currently in human mode
            if session.current_agent != AgentType.HUMAN:
                await manager.send_command_message(
                    {
                        "type": "command_result",
                        "action": "transfer_to_bot",
                        "success": False,
                        "message": "Session already handled by bot"
                    },
                    session_id
                )
                return
            
            # Transfer to bot
            transfer_message = await services.human_handler.transfer_to_bot(
                session_id,
                chat_history
            )
            
            # Update session agent type
            session.current_agent = AgentType.BOT
            
            # Notify the client that the transfer was successful
            await manager.send_command_message(
                {
                    "type": "command_result",
                    "action": "transfer_to_bot",
                    "success": True,
                    "message": transfer_message
                },
                session_id
            )
            await manager.broadcast_to_session(
                session_id,
                {
                    "type": "new_message",
                    "message": {
                        "role": "SYSTEM",
                        "content": transfer_message,
                        "timestamp": message_time.isoformat(),
                        "session_id": session_id,
                        "customer_id": customer_id
                    }
                }
            )
        else:
            # Unknown action
            await manager.send_command_message(
                {
                    "type": "command_result",
                    "action": action,
                    "success": False,
                    "message": f"Unknown action: {action}"
                },
                session_id
            )
    except Exception as e:
        logger.error(f"Error handling command {action}: {e}")
        await manager.send_command_message(
            {
                "type": "command_result",
                "action": action,
                "success": False,
                "message": f"Error: {str(e)}"
            },
            session_id
        )



@router.websocket("/chat/{session_id}/{client_type}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str,
    client_type: str, 
    services: ServiceContainer = Depends(get_websocket_service_container)
):
    """Handles WebSocket connections for real-time chat functionality.
    
    Function responsibilities:
        1. Manages real-time chat messaging via WebSocket connection
        2. Works alongside HTTP API services where:
        - API services handle: authentication, session creation, initial data
            retrieval
        - WebSocket handles: persistent real-time messaging
    
    Workflow:
        1. Accepts WebSocket connection request
        2. Connects client to session manager
        3. Retrieves and sends initial chat history
        4. Processes incoming messages in real-time loop
        5. Handles different message types based on client_type
    
    Args:
        websocket: The WebSocket connection object.
        session_id: Unique identifier for the chat session (created via API
            services).
        client_type: Type of client connecting ("customer" or "staff").
        services: Container with service dependencies for chat functionality.
    
    Raises:
        WebSocketDisconnect: When the client disconnects from the WebSocket.
        Exception: For any other errors that occur during WebSocket
            communication.
    """
    try:
        logger.info(f"WebSocket connection attempt - session: {session_id}, client: {client_type}")
        await websocket.accept()
        await manager.connect(websocket, session_id, client_type)
        # Get customer_id from query parameters
        query_params = dict(websocket.query_params)
        customer_id = query_params.get("customer_id", "")
        logger.info(f"Customer ID: {customer_id}")
        await services.get_or_create_session(session_id, customer_id)
        chat_history = await services.get_chat_history(session_id, customer_id)
        recent_messages = await chat_history.get_recent_turns(20)
        logger.info(f"Recent messages: {recent_messages}")

        # Format messages for the websocket response without session filtering
        formatted_messages = []
        for msg in recent_messages:
            formatted_messages.append({
                "role": msg.get("role", "SYSTEM"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now()).isoformat(),
                "customer_id": msg.get("customer_id", customer_id),
                "session_id": msg.get("session_id", session_id)  # Keep track of which session the message was from
            })
        logger.info(f"Formatted messages: {formatted_messages}")
        
        # Send the initial chat history to the newly connected client
        await websocket.send_json({
            "type": "history",
            "messages": formatted_messages
        })
        
        # Keep connection alive, handle client messages
        while True:
            # wait for incoming message from frontend
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "message":
                content = message_data.get("content")
                customer_id = message_data.get("customer_id", "")
                client_message_id = message_data.get("client_message_id", None)
                # Generate a single timestamp for consistency
                message_time = datetime.now()
                
                if client_type == "customer":
                    await handle_customer_message(
                        services, session_id, customer_id, content, manager,
                        message_time
                    )
                elif client_type == "staff":
                    await handle_staff_message(
                        services, session_id, customer_id, content, manager,
                        client_message_id, message_time
                    )
            elif (
                message_data.get("type") == "command" 
                and client_type == "staff"
            ):
                # Only staff can send commands
                action = message_data.get("action")
                customer_id = message_data.get("customer_id", "")
                await handle_command(
                    services, session_id, customer_id, action, manager,
                    message_time
                )
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id} - {client_type}")
        await manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011)  # 1011 = Internal Error
            except Exception as close_error:
                logger.error(f"Error closing WebSocket: {close_error}")
        await manager.disconnect(websocket, session_id)