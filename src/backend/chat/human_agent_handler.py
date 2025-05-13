"""Agent management, decides when to toggle to a human agent.
Sesstion management, agent switching, toggle(transfer) management
"""
from typing import Dict
import logging
import datetime
from src.backend.models.human_agent import (
    AgentType,
    ToggleReason,
)
from src.backend.chat.chat_history import ChatHistory
from src.backend.utils.llm import LLM

logger = logging.getLogger(__name__)


class HumanAgentHandler:
    def __init__(self, services):
        self.services = services
        self.cfg = services.cfg
        self.llm = LLM()
        self.prompts = self.cfg.human_agent_prompts

    async def _detect_human_request(
        self, query: str, recent_history: str
    ) -> bool:
        """Use LLM to detect if user is requesting human agent"""
        user_prompt = self.prompts.user_prompt.format(
            formatted_history=recent_history,
            query=query
        )
        response = await self.llm.generate(
            self.prompts.sys_prompt,
            user_prompt
        )
        logger.info(f"LLM Response: {response}")
        decision = response.split('\n')[0] 
        return decision.startswith('TRANSFER')
    
    async def transfer_to_human(
        self,
        session_id: str,
        reason: ToggleReason
    ) -> bool:
        """Transfer chat to human agent"""
        logger.info("Transfer chat to Human agent...")
        session = self.services.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        chat_history = await self.services.get_chat_history(
            session_id, session.customer_id)
        transfer_context = await chat_history.get_transfer_context()
        await chat_history.add_turn(
            'system',
            "Chat transferred to human agent",
            {
                'transfer_reason': reason.value,
                'transfer_context': transfer_context
            }
        )
        session.current_agent = AgentType.HUMAN
        return True

    async def transfer_to_bot(
        self,
        session_id: str,
        chat_history: ChatHistory
    ) -> str:
        """Transfer chat back to bot agent."""
        logger.info("Transfer chat back to Bot agent...")
        if session_id not in self.services.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.services.active_sessions[session_id]
        
        if session.current_agent == AgentType.BOT:
            return "Chat already handled by bot" 
        transfer_message = "Chat transferred back to AI assistant"
        await chat_history.add_turn(
            'system',
            transfer_message,
            {'reason': "agent_initiated"}
        )
        # Update session
        session.current_agent = AgentType.BOT
        session.last_interaction = datetime.datetime.now()
        return transfer_message

    async def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific chat session."""
        if session_id not in self.services.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.services.active_sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "customer_id": session.customer_id,
            "current_agent": session.current_agent.value,
            "session_duration": (datetime.datetime.now() - session.start_time).seconds,
            "last_interaction": session.last_interaction,
            "sentiment_score": session.sentiment_score,
            "transfer_history": session.transfer_history
        }

    def close_session(self, session_id: str) -> None:
        """Close and cleanup a chat session."""
        if session_id in self.services.active_sessions:
            del self.services.active_sessions[session_id]
