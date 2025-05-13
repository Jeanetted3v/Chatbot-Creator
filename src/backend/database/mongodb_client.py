import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import re
import asyncio
from pymongo.server_api import ServerApi

logger = logging.getLogger(__name__)


class MongoDBClient:
    def __init__(
        self,
        mongo_uri: str,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        timeout_params = "connectTimeoutMS=30000&socketTimeoutMS=30000&serverSelectionTimeoutMS=30000"
        # Set longer timeouts and add retryWrites option to connection string
        if "?" in mongo_uri:
            mongo_uri += "&" + timeout_params
        else:
            mongo_uri += "?" + timeout_params
        
        self.uri = mongo_uri
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        
    async def connect(self) -> None:
        """Establish connection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.client = AsyncIOMotorClient(
                    self.uri,
                    connect=True,  # Force initial connection
                    serverSelectionTimeoutMS=30000,
                    server_api=ServerApi('1')
                )
                
                # Test the connection
                await self.test_connection()
                logger.info("Successfully connected to MongoDB")
                return
                
            except (ServerSelectionTimeoutError, ConnectionFailure) as e:
                if attempt == self.max_retries - 1:
                    logger.info("Failed to connect after "
                                f"{self.max_retries} attempts")
                    raise
                logger.info(f"Connection attempt {attempt + 1} failed: {str(e)}")
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                
    async def test_connection(self) -> None:
        """Test MongoDB connection"""
        if not self.client:
            raise ConnectionError("Client not initialized")
            
        try:
            await self.client.admin.command('ping')
        except Exception as e:
            logger.info(f"Connection test failed: {str(e)}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.client:
            self.client.close()
            self.client = None