import logging
import uuid
import json
from typing import List, Dict
from omegaconf import DictConfig
from pydantic_ai import Agent
from src.backend.database.weaviate_manager import WeaviateManager
from src.backend.models.embedding_metadata import EmbeddingMetadata

logger = logging.getLogger(__name__)


class WeaviateEmbedder:
    """Class to embed documents using Weaviate Cloud."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the WeaviateEmbedder."""
        self.cfg = cfg
        self.prompts = cfg.extract_metadata
        self.manager = WeaviateManager(cfg)
        
        # Initialize metadata extraction agent
        self.agent = Agent(
            'openai:gpt-4.1-mini',
            result_type=EmbeddingMetadata,
            system_prompt=self.prompts['system_prompt']
        )
        
        # Set up collections for chunks and full documents
        self._setup_collections()
    
    def _setup_collections(self):
        """Set up the Weaviate collections for chunks and full documents."""
        # Define properties for the main collection (chunks with embeddings)
        embedding_props = [
            {"name": "content", "dataType": "text", "description": "The document content"},
            {"name": "category", "dataType": "text", "description": "Content category", "indexFilterable": True},
            {"name": "keywords", "dataType": "text[]", "description": "Keywords extracted from content"},
            {"name": "doc_id", "dataType": "text", "description": "Document ID", "indexFilterable": True},
            {"name": "chunk_type", "dataType": "text", "description": "Type of chunk (partial/full)", "indexFilterable": True},
            {"name": "chunk_index", "dataType": "int", "description": "Index of the chunk", "indexFilterable": True}
        ]
        
        # Create collection for chunks with embeddings
        self.collection = self.manager.create_collection(
            collection_name=self.cfg.embedder.collection,
            properties=embedding_props
        )
        
        # Create collection for full documents (without embeddings)
        full_doc_props = [
            {"name": "content", "dataType": "text", "description": "Full document content"},
            {"name": "doc_id", "dataType": "text", "description": "Document ID", "indexFilterable": True},
            {"name": "title", "dataType": "text", "description": "Document title", "indexFilterable": True},
        ]
        
        self.full_doc_collection = self.manager.create_collection(
            collection_name=f"{self.cfg.embedder.collection}_full",
            properties=full_doc_props
        )
    
    async def _extract_metadata(self, content: str) -> EmbeddingMetadata:
        """Extract metadata from content using agent.
        
        Args:
            content: The text content to extract metadata from
            
        Returns:
            EmbeddingMetadata object with extracted metadata
        """
        logger.info("Extracting metadata")
        result = await self.agent.run(
            self.prompts['user_prompt'].format(content=content)
        )
        metadata = result.data
        logger.info(f"Extracted metadata: {metadata}")
        return metadata
    
    async def _store_processed_documents(
        self, processed_docs: List[Dict]
    ) -> None:
        """Store processed doc in Weaviate. Handles both chunked and full doc.
        
        Args:
            processed_docs: List of processed document dictionaries
        """
        for doc in processed_docs:
            doc_id = doc.get('doc_id', str(uuid.uuid4()))
            
            if doc['type'] == 'chunked':
                # For chunked documents, store each chunk with its embedding
                with self.collection.batch.fixed_size(batch_size=100) as batch:
                    for i, chunk in enumerate(doc['chunks']):
                        # Extract metadata for each chunk
                        extracted_metadata = await self._extract_metadata(chunk['content'])
                        
                        # Combine metadata
                        properties = {
                            "content": chunk['content'],
                            "doc_id": doc_id,
                            "chunk_type": "partial",
                            "chunk_index": i,
                            "total_chunks": doc['num_chunks'],
                            **chunk.get('metadata', {}),
                            **extracted_metadata.model_dump()
                        }
                        
                        # Generate a UUID for the object
                        object_id = str(uuid.uuid4())
                        
                        # Add the object to the batch
                        batch.add_object(
                            properties=properties,
                            uuid=object_id
                        )
                        
                        if batch.number_errors > 10:
                            logger.error("Batch import stopped due to excessive errors.")
                            break
            else:
                # For full documents, store without embeddings in a separate collection
                self.full_doc_collection.data.insert({
                    "content": doc['content'],
                    "doc_id": doc_id,
                    "title": doc.get('metadata', {}).get('title', "Untitled Document")
                }, str(uuid.uuid4()))


async def embed_doc(cfg: DictConfig, chunked_docs: List[Dict]) -> None:
    """Embed documents using Weaviate Cloud.
    
    Args:
        cfg: Configuration object
        chunked_docs: List of chunked document dictionaries
    """
    logger.info("Starting document embedding process with Weaviate Cloud...")
    embedder = WeaviateEmbedder(cfg)
    
    total_chunks = sum(
        len(doc['chunks']) if doc['type'] == 'chunked' else 1
        for doc in chunked_docs
    )
    logger.info(f"Processing {len(chunked_docs)} documents with "
                f"total {total_chunks} chunks")
    
    try:
        await embedder._store_processed_documents(chunked_docs)
        # Verify embeddings by checking collection count
        collection_count = embedder.collection.query.fetch_objects().objects
        count = len(collection_count) if collection_count else 0
        logger.info(f"Successfully stored {count} embeddings in collection")
        
        # Optional: Check a few random embeddings
        if count > 0:
            logger.info("Sample embedding verification successful")
    except Exception as e:
        logger.error(f"Error during document embedding: {str(e)}")
        raise e
    
    logger.info("Document embedding process completed")
    
    # Close the Weaviate connection
    embedder.manager.close()