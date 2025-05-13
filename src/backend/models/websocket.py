# WebSocket message models
class WSMessage(BaseModel):
    """Base model for WebSocket messages"""
    type: str
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class WSChatMessage(WSMessage):
    """WebSocket message containing chat content"""
    type: str = "message"
    role: MessageRole
    content: str
    customer_id: str
    metadata: Optional[Dict[str, Any]] = None

class WSSystemNotification(WSMessage):
    """WebSocket notification for system events"""
    type: str = "system"
    message: str
    event: str  # 'handover', 'transfer_to_bot', 'error', etc.
    data: Optional[Dict[str, Any]] = None
