from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class MessageRole(str, Enum):
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"
    HUMAN_AGENT = "human_agent"


class AgentType(str, Enum):
    BOT = "bot"
    HUMAN = "human"


class ChatTurn(BaseModel):
    """Model for a single message in the conversation"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    customer_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatSession(BaseModel):
    session_id: str
    customer_id: str
    current_agent: AgentType = AgentType.BOT
    start_time: datetime = Field(default_factory=datetime.now)
    last_interaction: datetime = Field(default_factory=datetime.now)
    sentiment_score: float = 1.0
    sentiment_confidence: float = 1.0
    message_count: int = 0
    last_analyzed_msg_index: int = 0


class ToggleReason(Enum):
    CUSTOMER_REQUEST = "customer_request"
    AGENT_INITIATED = "agent_initiated"
    SENTIMENT_BASED = "sentiment_based"


class AgentDecision(BaseModel):
    """Used in HumanAgentHandler, process_message method"""
    should_transfer: bool
    response: Optional[str]
    transfer_reason: Optional[ToggleReason] = None


class AnalysisResult(BaseModel):
    """for Message analyzer class"""
    score: float
    confidence: float
    method_used: str
    full_analysis: bool
    triggers_detected: list[str] = None
    analysis_details: Dict = None
