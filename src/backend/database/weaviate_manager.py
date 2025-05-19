import logging
import os
from typing import Dict, Optional, List, Any
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
from utils.settings import SETTINGS

logger = logging.getLogger(__name__)


class WeaviateManager:
    """Class to manage Weaviate connections and operations."""
    
    def __init__(self, cfg):
        """Initialize the Weaviate manager."""
        self.cfg = cfg
        self.client = self._create_client()
        
    def _create_client(self):
        """Create and return a Weaviate client."""
        # Set up API credentials
        weaviate_url = SETTINGS.WEAVIATE_ENDPOINT
        weaviate_key = SETTINGS.WEAVIATE_API_KEY
        
        # Set up Azure OpenAI credentials if using Azure
        if self.cfg.weaviate.embedding_provider.lower() == "azure":
            azure_key = SETTINGS.AZURE_API_KEY
            headers = {"X-Azure-Api-Key": azure_key}
        else:
            # For OpenAI or other providers
            headers = {}
        
        # Connect to Weaviate
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_key),
                headers=headers
            )
            logger.info(f"Successfully connected to Weaviate at {weaviate_url}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise
    
    def create_collection(
        self,
        collection_name: str,
        properties: List[Dict],
    ) -> Any:
        """Create a collection in Weaviate.
        
        Args:
            collection_name: Name of the collection
            properties: List of property definitions
            
        Returns:
            Weaviate collection object
        """
        try:
            # Check if collection exists and delete if specified in config
            if (
                self.collection_exists(collection_name)
                and self.cfg.weaviate.get('recreate_collections', False)
            ):
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.collections.delete(collection_name)
            
            # Configure the vectorizer based on the provider
            if self.cfg.weaviate.embedding_provider.lower() == "azure":
                vectorizer = [
                    Configure.NamedVectors.text2vec_azure_openai(
                        name="default",
                        source_properties=["content"] + [
                            prop['name'] for prop in properties
                            if prop.get('vectorize', True)
                        ],
                        resource_name=self.cfg.llm.azure_resource_name,
                        deployment_id=self.cfg.llm.azure_deployment_id,
                    )
                ]
            elif self.cfg.weaviate.embedding_provider.lower() == "openai":
                vectorizer = [
                    Configure.NamedVectors.text2vec_openai(
                        name="default",
                        source_properties=["content"] + [
                            prop['name'] for prop in properties
                            if prop.get('vectorize', True)
                        ],
                        model_name=self.cfg.llm.embedding_model,
                    )
                ]
            else:
                raise ValueError(f"Unsupported provider: {self.cfg.weaviate.embedding_provider}")
            
            # Create the collection properties
            property_defs = []
            for prop in properties:
                prop_config = {
                    "name": prop['name'],
                    "dataType": prop.get('dataType', 'text'),
                    "description": prop.get('description', f"Property {prop['name']}"),
                }
                # Only add optional parameters if they exist
                for key in ['tokenization', 'indexFilterable', 'indexSearchable']:
                    if key in prop:
                        prop_config[key] = prop[key]
                
                property_defs.append(prop_config)
            
            # Create the collection
            collection = self.client.collections.create(
                name=collection_name,
                vectorizer_config=vectorizer,
                properties=property_defs,
            )
            
            logger.info(f"Created collection: {collection_name}")
            return collection
        
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Weaviate.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.collections.list_all()
            return collection_name in [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            return False
    
    def get_collection(self, collection_name: str) -> Any:
        """Get a collection from Weaviate.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Weaviate collection object or None if not found
        """
        try:
            if self.collection_exists(collection_name):
                return self.client.collections.get(collection_name)
            logger.warning(f"Collection {collection_name} does not exist")
            return None
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {str(e)}")
            return None
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate client connection closed")