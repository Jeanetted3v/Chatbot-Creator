from fastapi import WebSocket
import logging
from typing import Dict
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}

    # Structure: {session_id: {client_id: websocket}}
    async def connect(self, websocket, session_id, client_id):   # client_id is customer or staff
        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}
        self.active_connections[session_id][client_id] = websocket

    async def disconnect(self, websocket, session_id):
        """Disconnect a WebSocket and clean up."""
        try:
            # Find and remove the connection
            if session_id in self.active_connections:
                for client_id, conn in list(
                    self.active_connections[session_id].items()
                ):
                    if conn == websocket:
                        del self.active_connections[session_id][client_id]
                        logger.info(
                            f"Removed connection {client_id} from session "
                            f"{session_id}"
                        )
                        break
                
                # If no more connections for this session, remove the session entry
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
                    logger.info(f"Removed empty session {session_id}")
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")

    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            closed_connections = []
            for client_id, websocket in self.active_connections[session_id].items():
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        f"Error broadcasting message to {client_id}: {e}"
                    )
                    closed_connections.append(client_id)
            # Clean up closed connections
            for client_id in closed_connections:
                try:
                    del self.active_connections[session_id][client_id]
                    logger.info(f"Removed closed connection for {client_id} "
                                "in session {session_id}")
                except KeyError:
                    pass
    
    async def send_command_message(self, message, session_id):
        """Send a message to all connections in a session.
        
        Different from broadcast_to_session:
            - Broadcast_to_session is typically used for general chat messages
                that should be seen by everyone
            - Send_command_message is typically used for command responses,
                status updates, or other administrative messages that are
                specific to a particular chat session
        send back the result of a command operation (like "takeover successful"
        or "transfer to bot complete") to the clients connected to that
        specific session.
        """
        if session_id in self.active_connections:
            for websocket in self.active_connections[session_id].values():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending command message: {e}")


manager = ConnectionManager()