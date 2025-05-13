import logging
from typing import Tuple, Optional, List
from pydantic import BaseModel
from pydantic_ai import Agent
from src.backend.models.human_agent import (
    AgentDecision,
    ToggleReason,
    AgentType,
    MessageRole
)
from src.backend.utils.llm_model_factory import LLMModelFactory

logger = logging.getLogger(__name__)


class ReasongingResult(BaseModel):
    """Result model for reasoning agent"""
    expanded_query: List[str]
    need_search: bool


class ResponseResult(BaseModel):
    """Response model for query handler"""
    response: str
    intent: str
    rag_result_used: Optional[str]
    english_level: Optional[str]
    course_interest: Optional[str]
    lexile_level: Optional[str]


class QueryHandler:
    def __init__(self, services):
        """Initialize QueryHandler with all class instances from services"""
        self.services = services
        self.cfg = services.cfg
        self.mongo_client = services.mongodb_client.client
        reason_model_config = dict(self.cfg.reasoning)
        reasoning_model = LLMModelFactory.create_model(reason_model_config)
        self.reasoning_agent = Agent(
            model=reasoning_model,
            result_type=ReasongingResult,
            system_prompt=self.cfg.query_handler_prompts.reasoning_agent['sys_prompt']
        )
        response_model_config = dict(self.cfg.response)
        response_model = LLMModelFactory.create_model(response_model_config)
        self.response_agent = Agent(
            model=response_model,
            result_type=ResponseResult,
            system_prompt=self.cfg.query_handler_prompts.response_agent['sys_prompt']

        )
    
    async def analyze_sentiment(
        self, session_id: str, customer_id: str, message: str, total_count: int
    ) -> Tuple[dict, AgentDecision]:
        session = self.services.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None, AgentDecision(
                should_transfer=False,
                response="Session not found. Please try again.",
                transfer_reason=None
            )
        # Get chat history for stats
        chat_history = await self.services.get_chat_history(
            session_id, customer_id)

        # If already with human agent, continue there
        if session.current_agent == AgentType.HUMAN:
            await chat_history.add_turn('user', message)  # Add message without analysis
            return {}, AgentDecision(
                should_transfer=True,
                response=None,  # Let human agent UI handle response
                transfer_reason=None
            )
        recent_turns = await chat_history.get_recent_turns()
        last_analyzed = sum(1 for turn in recent_turns if turn.get(
            'full_analysis', False)
        )

        # Default values if we skip analysis
        analysis_result = None
        agent_decision = AgentDecision(
            should_transfer=False,
            response=None,
            transfer_reason=None
        )
        if total_count < self.cfg.msg_analyzer.min_message_length:
            logger.info("Not enough messages for sentiment analysis")
            return analysis_result, agent_decision
        elif not self.services.message_analyzer:
            logger.info("Message analyzer is None, skipping sentiment analysis")
            analysis_result = None
            return analysis_result, agent_decision
        else:
            analysis_result = await self.services.message_analyzer.analyze(
                message,
                total_count,
                last_analyzed
            )
            logger.info(f"Analysis result: {analysis_result}")
        # Check if human agent is needed
        recent_history = await chat_history.format_history_for_prompt()
        needs_human = await self.services.human_handler._detect_human_request(
            message, recent_history
        )
        logger.info(f"Needs human agent: {needs_human}")
        
        # Determine if we should transfer
        should_transfer = (
            needs_human or
            (
                analysis_result.score <
                self.services.human_handler.cfg.human_agent.sentiment_threshold and
                analysis_result.confidence >
                self.services.human_handler.cfg.human_agent.confidence_threshold
            )
        )
        
        # Handle transfer if needed
        if should_transfer:
            transfer_reason = (
                ToggleReason.CUSTOMER_REQUEST if needs_human
                else ToggleReason.SENTIMENT_BASED
            )
            # Return decision, let the caller handle the transfer
            return analysis_result, AgentDecision(
                should_transfer=True,
                response="Transferring to human agent...",
                transfer_reason=transfer_reason
            )
        
        return analysis_result, agent_decision

    async def handle_query(
        self, query: str, session_id: str, customer_id: str
    ) -> tuple[str, dict, str]:
        """Main entry point for handling user queries."""
        try:
            session = await self.services.get_or_create_session(
                session_id, customer_id
            )
            chat_history = await self.services.get_chat_history(
                session_id, customer_id)

            if session.current_agent == AgentType.HUMAN:
                # Add message to chat history without any sentiment analysis
                await chat_history.add_turn(MessageRole.USER, query)
                logger.info(f"Message from customer forwarded to human agent "
                            f"for session {session_id}")
                return "Message forwarded to human agent"

            # For bot processing:
            # Step 1: Analyze sentiment and check if should transfer to human
            total_count = await chat_history.collection.count_documents(
                {"session_id": session_id})
            logger.info(f"Current session message count: {total_count}")
            analysis_result, agent_decision = await self.analyze_sentiment(   # need to handle is msg analyzer is None
                session_id, customer_id, query, total_count)
            
            # Add message to chat history with metadata from analysis
            if analysis_result:
                metadata = {
                    'sentiment_score': analysis_result.score,
                    'sentiment_confidence': analysis_result.confidence,
                    'full_analysis': analysis_result.full_analysis
                }
                await chat_history.add_turn(
                    MessageRole.USER, query, metadata=metadata
                )
            else:
                await chat_history.add_turn(MessageRole.USER, query)
            logger.info(f"Analysis result: {analysis_result}")
            logger.info(f"AgentDecision if transfer to human: {agent_decision}")
            
            if agent_decision.should_transfer:
                success = await self.services.human_handler.transfer_to_human(
                    session_id,
                    agent_decision.transfer_reason
                )
                logger.info(f"Transfer to human agent: {success}")
                if success:
                    if agent_decision.response:
                        await chat_history.add_turn(
                            MessageRole.SYSTEM, agent_decision.response
                        )
                    return "Message forwarded to human agent"
                else:
                    transfer_failed_msg = (
                        "All our staff are currently busy. "
                        "I'll continue to assist you."
                    )
                    await chat_history.add_turn(
                        MessageRole.SYSTEM, transfer_failed_msg
                    )
                    return transfer_failed_msg

            msg_history = await chat_history.format_history_for_prompt()

            reasoning_result = await self.reasoning_agent.run(
                self.cfg.query_handler_prompts.reasoning_agent['user_prompt'].format(
                    query=query,
                    message_history=msg_history,
                    competitors=self.cfg.guardrails.competitors,
                ),
            )
            logger.info(f"Reasoning result: {reasoning_result.data}")
            need_search = reasoning_result.data.need_search
            
            if need_search:
                all_search_results = []
                for eq in reasoning_result.data.expanded_query:
                    search_results = await self.services.hybrid_retriever.search(eq)
                    all_search_results.append(search_results)
                logger.info(f"All search results: {all_search_results}")
            else:
                all_search_results = []

            result = await self.response_agent.run(
                self.cfg.query_handler_prompts.response_agent['user_prompt'].format(
                    query=query,
                    message_history=msg_history,
                    search_results=all_search_results,
                    competitors=self.cfg.guardrails.competitors,
                )
            )
            response = result.data.response
            intent = result.data.intent
            rag_result_used = result.data.rag_result_used
            logger.info(f"Intent: {intent}, "
                        f"RAG Result used: {rag_result_used}"
                        f", Response: {response}")
            await chat_history.add_turn(
                MessageRole.BOT,
                response,
                metadata={
                    'intent': intent,
                    'rag_result_used': rag_result_used
                }
            )
            return response
                
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            error_msg = (
                "Sorry, there was an error processing your query. "
                "Please try again later or contact our support."
            )
            await chat_history.add_turn(MessageRole.BOT, error_msg)
            return error_msg