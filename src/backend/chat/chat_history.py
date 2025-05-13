import logging
from datetime import datetime
from typing import List, Dict, Optional
from pymongo import DESCENDING
from src.backend.models.human_agent import ChatTurn, MessageRole


logger = logging.getLogger(__name__)


class ChatHistory:
    def __init__(
        self,
        cfg,
        session_id: str,
        customer_id: str,
        collection=None,
        max_turns_for_prompt: int = 30
    ):
        self.cfg = cfg
        self.collection = collection
        self.session_id = session_id
        self.customer_id = customer_id
        self.conversation_turns = []
        self.max_turns_for_prompt = max_turns_for_prompt
        self._last_processed_timestamp = None  # Track last processed message

    async def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a turn to conversation history with full metadata for MongoDB"""
        timestamp = datetime.now()
        try:
            role_str = (
                role.value
                if isinstance(role, MessageRole)
                else str(role).lower()
            )
            turn = ChatTurn(
                role=role_str,
                content=content,
                timestamp=timestamp,
                customer_id=self.customer_id,
                session_id=self.session_id,
                metadata=metadata
            )
            self.conversation_turns.append(turn)
            turn_dict = (
                turn.dict()
                if hasattr(turn, 'dict')
                else turn.model_dump()
            )
            result = await self.collection.insert_one(turn_dict)
            logger.info(f"Added message with ID: {result.inserted_id}")
            
            message_data = {
                "type": "new_message",
                "message": {
                    "role": role_str,
                    "content": content,
                    "timestamp": timestamp.isoformat(),
                    "customer_id": self.customer_id,
                    "session_id": self.session_id
                }
            }
            from src.backend.api.websocket_manager import manager
            await manager.broadcast_to_session(self.session_id, message_data)
        except Exception as e:
            logger.error(f"Error adding turn to history: {str(e)}")

    async def format_history_for_prompt(self) -> str:
        """Format last N turns in simple format for prompt to save tokens"""
        turns = []
        try:
            cursor = self.collection.find({
                'customer_id': self.customer_id,
                'session_id': self.session_id
            }).sort('timestamp', DESCENDING).limit(self.max_turns_for_prompt)
            
            turns = await cursor.to_list(length=self.max_turns_for_prompt)
            
            if not turns:
                logger.info("No messages found with customer_id, session_id. "
                            "Trying without filters...")
                cursor = self.collection.find({}).sort(
                    'timestamp', DESCENDING
                ).limit(self.max_turns_for_prompt)
                turns = await cursor.to_list(length=self.max_turns_for_prompt)
            
            # Reverse to get chronological order
            turns.reverse()
            
            # Format the turns into a string
            formatted_history = "\n".join(
                f"{turn.get('role', 'UNKNOWN').capitalize()}: "
                f"{turn.get('content', '')}"
                for turn in turns
            )
            logger.info(f"Successfully retrieved {len(turns)} messages")
            return formatted_history
        except Exception as e:
            error_msg = f"Error retrieving conversation history: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def get_recent_turns(self, limit: int = 10) -> List[ChatTurn]:
        """Get recent turns from MongoDB"""
        try:
            # Try with customer_id and session_id first
            cursor = self.collection.find({
                'customer_id': self.customer_id
            }).sort('timestamp', DESCENDING).limit(limit)
            
            turns = await cursor.to_list(length=limit)
            logger.info(f"Retrieved {len(turns)} messages for customer {self.customer_id}")
            
            # If still no results, fall back to no filters (for backward compatibility)
            # if not turns:
            #     cursor = self.collection.find({}).sort('timestamp', DESCENDING).limit(limit)
            #     turns = await cursor.to_list(length=limit)
            return turns
            
        except Exception as e:
            logger.error(f"Error getting recent turns: {str(e)}")
            return []

    def get_full_history(self) -> List[Dict]:
        """Get full conversation history for MongoDB storage"""
        return self.conversation_turns
        
    async def get_transfer_context(self) -> Dict:
        """Get context information when transferring to human agent
        
        Being called in HumanAgentHandler.transfer_to_human method
        """
        try:
            recent_turns = await self.get_recent_turns(10)
            sentiments = []
            for turn in recent_turns:
                metadata = turn.get('metadata')
                if metadata is not None:
                    sentiment_score = metadata.get('sentiment_score')
                    if sentiment_score is not None:
                        sentiments.append(sentiment_score)
            logger.info(f"Sentiment from get_transfer_context: {sentiments}")
            transfer_context = {
                'recent_conversation': recent_turns,
                'sentiment_trend': sentiments,
                'total_turns': len(recent_turns),
                'conversation_start': (
                    recent_turns[0]['timestamp'] if recent_turns else None
                )
            }
            logger.info(f"[ChatHistory] Transfer Context: {transfer_context}")
            return transfer_context
        except Exception as e:
            logger.error(f"Error getting transfer context: {str(e)}")
            return {}
        