from datetime import datetime
import logging
from uuid import uuid4
from src.backend.database.mongodb_client import MongoDBClient
from src.backend.chat.hybrid_retriever import HybridRetriever
from src.backend.chat.sentiment_analyzer import SentimentAnalyzer
from src.backend.chat.msg_analyzer import MessageAnalyzer
from src.backend.chat.human_agent_handler import HumanAgentHandler
from src.backend.chat.query_handler import QueryHandler
from src.backend.chat.chat_history import ChatHistory
from src.backend.models.human_agent import AgentType, ChatSession
from src.backend.utils.settings import SETTINGS


logger = logging.getLogger(__name__)


class ServiceContainer:
    """Container for all service instances with centralized initialization."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.mongodb_client = None
        self.db = None
        self.sessions_collection = None
        self.chat_history_collection = None
        self.hybrid_retriever = None
        self.sentiment_analyzer = None
        self.message_analyzer = None
        self.human_handler = None
        self.query_handler = None
        self.chat_histories = {}
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize all service components with proper dependency order."""
        try:
            self.mongodb_client = MongoDBClient(SETTINGS.MONGODB_URI)
            await self.mongodb_client.connect()
            self.db = self.mongodb_client.client[self.cfg.mongodb.db_name]
            self.chat_history_collection = self.db[
                self.cfg.mongodb.chat_history_collection
            ]
            self.sessions_collection = self.db[
                self.cfg.mongodb.session_collection
            ]
            self.hybrid_retriever = HybridRetriever(self.cfg)
            self.sentiment_analyzer = SentimentAnalyzer(self.cfg)
            self.message_analyzer = MessageAnalyzer(self)
            self.human_handler = HumanAgentHandler(self)
            self.query_handler = QueryHandler(self)
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise
    
    async def get_chat_history(self, session_id: str, customer_id: str):
        """Get or create chat history for a session."""
        if session_id not in self.chat_histories:
            logger.info(f"Creating new chat history for {session_id}")
            self.chat_histories[session_id] = ChatHistory(
                self.cfg,
                session_id,
                customer_id,
                collection=self.chat_history_collection
            )
        return self.chat_histories[session_id]
    
    async def get_or_create_session(
        self,
        session_id: str,
        customer_id: str
    ) -> ChatSession:
        """Get existing session or create a new one"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_interaction = datetime.now()
            # Update in MongoDB if client exists
            if self.mongodb_client and self.mongodb_client.client:
                await self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {"last_interaction": datetime.now()}}
                )
            return session
        
        # Check if session exists in MongoDB
        if self.mongodb_client and self.mongodb_client.client:
            db_session = await self.sessions_collection.find_one(
                {"session_id": session_id}
            )
            if db_session:
                # Create session from MongoDB data
                session = ChatSession(
                    session_id=db_session["session_id"],
                    customer_id=db_session["customer_id"],
                    current_agent=db_session.get("current_agent", "bot").lower(),
                    start_time=db_session.get("start_time", datetime.now()),
                    last_interaction=datetime.now(),
                    message_count=db_session.get("message_count", 0)
                )
                self.active_sessions[session_id] = session
                # Update last interaction in MongoDB
                await self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {"last_interaction": datetime.now()}}
                )
                
                logger.info(f"Loaded existing session {session_id} in database")
                return session

        session = ChatSession(
            session_id=session_id,
            customer_id=customer_id,
            current_agent=AgentType.BOT,
            start_time=datetime.now(),
            last_interaction=datetime.now()
        )
        self.active_sessions[session_id] = session
        # Persist to MongoDB if client exists
        if self.mongodb_client and self.mongodb_client.client:
            session_data = {
                "session_id": session_id,
                "customer_id": customer_id,
                "current_agent": "BOT",  # Convert enum to string for MongoDB
                "start_time": datetime.now(),
                "last_interaction": datetime.now(),
                "message_count": 0
            }
            await self.sessions_collection.insert_one(session_data)
        
        logger.info(f"Created new session {session_id} for "
                    f"customer {customer_id}")
        return session
    
    async def check_session(self, customer_id: str) -> str:
        """Check if a recent active session exists for a customer.
        
        Returns existing session_id if found, otherwise creates a new one.
        """
        # Get timeout from config or use default
        session_timeout_hours = (
            self.cfg.mongodb.timeout_hours
            if hasattr(self.cfg.mongodb, 'timeout_hours')
            else 24
        )
        
        # First check MongoDB for recent sessions
        if self.mongodb_client and self.mongodb_client.client:
            db_session = await self.sessions_collection.find_one(
                {"customer_id": customer_id},
                sort=[("last_interaction", -1)] # Sort by last interaction in descending order
            )
            
            if db_session:
                last_interaction = db_session.get("last_interaction")
                if last_interaction:
                    hours_since_last = (
                        (datetime.now() - last_interaction)
                        .total_seconds() / 3600
                    )
                    if hours_since_last < session_timeout_hours:
                        session_id = db_session["session_id"]
                        logger.info(
                            f"Found recent session {session_id} in database "
                            f" for customer {customer_id}"
                        )
                        return session_id
        
        # if no valid session is found, check active sessions in memory
        for session_id, session in self.active_sessions.items():
            if session.customer_id == customer_id:
                hours_since_last = (
                    (datetime.now() - session.last_interaction)
                    .total_seconds() / 3600
                )
                if hours_since_last < session_timeout_hours:
                    logger.info(
                    f"Found recent active session {session_id} "
                    f"for customer {customer_id}"
                    )
                    return session_id
        
        # Create new session ID if none found
        new_session_id = f"session-{uuid4()}"
        logger.info(f"No recent session found, creating new session ID "
                    f"{new_session_id} for customer {customer_id}")
        return new_session_id

    async def create_new_session(self, customer_id: str = None) -> dict:
        """Create a new session and return its details.
        
        If customer_id is provided, checks for existing recent sessions.
        """
        # Generate customer ID if not provided (for demo purposes)
        if not customer_id:
            customer_id = f"customer-{uuid4().hex[:8]}"
        session_id = await self.check_session(customer_id)
        session = await self.get_or_create_session(session_id, customer_id)
        return {
            "session_id": session_id,
            "customer_id": customer_id,
            "current_agent": session.current_agent,
            "start_time": session.start_time,
            "last_interaction": session.last_interaction,
            "message_count": 0
        }

    async def cleanup(self):
        """Cleanup all resources."""
        if self.mongodb_client:
            await self.mongodb_client.cleanup()
        # Clear dictionaries
        self.active_sessions.clear()
        self.chat_histories.clear()
        logger.info("Cleanup complete")


# TODO: Implement the following optuional methods:
# def get_sessions_needing_help(self):
#     """Get list of sessions that need human help"""
#     from models import AgentType
    
#     return [
#         session_id for session_id, session in self.active_sessions.items()
#         if getattr(session, 'needs_human_help', False) and 
#            session.current_agent == AgentType.BOT
#     ]

# def mark_session_needs_help(self, session_id, needs_help=True):
#     """Mark a session as needing human help"""
#     if session_id in self.active_sessions:
#         session = self.active_sessions[session_id]
#         session.needs_human_help = needs_help