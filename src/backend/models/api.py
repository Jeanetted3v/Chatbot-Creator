from pydantic import BaseModel, Field
from src.backend.models.human_agent import AgentType, MessageRole
from datetime import datetime
from typing import Optional


# Additional API models using your existing models as base
class MessageRequest(BaseModel):
    """Request model for sending a message"""
    message: str
    session_id: Optional[str] = None  # Optional - will be generated if not provided
    customer_id: str
    source: str = "api"  # 'api', 'whatsapp', 'web', etc.

class MessageResponse(BaseModel):
    """Response model for message API"""
    message: str
    session_id: str
    customer_id: str
    role: MessageRole
    current_agent: AgentType
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionResponse(BaseModel):
    """API response model for session info"""
    session_id: str
    customer_id: str
    current_agent: AgentType
    start_time: datetime
    last_interaction: datetime
    message_count: int


# Models for staff API
class StaffMessageRequest(BaseModel):
    """Request model for staff sending a message"""
    session_id: str
    customer_id: str
    message: str


class TransferRequest(BaseModel):
    """Request model for transferring back to bot"""
    session_id: str
    customer_id: str
    message: Optional[str] = None


class TakeoverRequest(BaseModel):
    """Request model for staff taking over a conversation"""
    session_id: str
    customer_id: str
    message: Optional[str] = None  # Optional custom message


