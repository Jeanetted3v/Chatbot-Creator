import logging
from typing import List, Dict, Optional
import uuid
import os
import json
from omegaconf import DictConfig
from pydantic_ai import Agent
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from utils.settings import SETTINGS
from src.backend.models.embedding_metadata import EmbeddingMetadata

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, cfg, client):
        self.client = client
        self.collection = None
        self.prompts = cfg.extract_metadata
        self.embedding_fn = None
        self.agent = Agent(
            'openai:gpt-4.1-mini',
            result_type=EmbeddingMetadata,
            system_prompt=self.prompts['system_prompt']
        )

    @classmethod
    async def create_chromadb_client(
        cls, cfg, chroma_host="localhost", chroma_port=8000
    ):
        client = await chromadb.AsyncHttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(anonymized_telemetry=False)
        )
        return cls(cfg, client)

    def _create_embedding_function(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None
    ) -> embedding_functions.EmbeddingFunction:
        if provider.lower() == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name
            )
        raise ValueError(f"Unsupported embedding provider: {provider}")

    async def _get_or_create_collection(
        self, collection_name: str, similarity_metric: str
    ):
        """Get existing collection or create a new one"""
        metadata = {"hnsw:space": similarity_metric}
        
        # Try to get collection first
        try:
            collection = await self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Using existing collection: {collection_name}")
            return collection
        except ValueError as e:
            # Handle embedding function mismatch by recreating collection
            if "Embedding function name mismatch" in str(e):
                logger.warning(f"Embedding function mismatch. Recreating collection: {collection_name}")
                try:
                    await self.client.delete_collection(collection_name)
                except Exception:
                    pass  # Collection might not exist, ignore error
            
            # Create new collection with embedding function
            return await self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Collection not found, creating new: {e}")
            return await self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata=metadata
            )
    
    async def _extract_metadata(self, content: str) -> EmbeddingMetadata:
        result = await self.agent.run(
            self.prompts['user_prompt'].format(content=content)
        )
        return result.data

    def _convert_metadata_str(self, metadata: Dict) -> Dict:
        return {
            key: json.dumps(value) if not isinstance(
                value,
                (str, int, float, bool)
            )
            else value
            for key, value in metadata.items()
        }

    async def _store_chunk(self, chunk, doc_id, total_chunks):
        """Process and store a single chunk."""
        # Extract metadata
        extracted_metadata = await self._extract_metadata(chunk['content'])
        
        # Prepare metadata
        enhanced_metadata = {
            **chunk['metadata'],
            **extracted_metadata.model_dump(),
            'chunk_type': 'partial',
            'total_chunks': total_chunks,
            'doc_id': doc_id
        }
        metadata = self._convert_metadata_str(enhanced_metadata)
        
        # Get embedding
        embedding = self.embedding_fn(chunk['content'])
        # Fix nested list issue if present
        if isinstance(embedding, list) and len(embedding) == 1:
            embedding = embedding[0]
            
        # Store in collection
        await self.collection.add(
            documents=[chunk['content']],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )
        
        return metadata

    async def _store_processed_documents(self, processed_docs: List[Dict]):
        """
        Store processed documents in ChromaDB.
        Handles both chunked and full documents appropriately.
        """
        for doc in processed_docs:
            if doc['type'] == 'chunked':
                # Get doc ID
                doc_id = doc.get('doc_id', str(uuid.uuid4()))
                total_chunks = doc['num_chunks']
                batch_size = 5
                for i in range(0, len(doc['chunks']), batch_size):
                    batch = doc['chunks'][i:i+batch_size]
                    for chunk in batch:
                        try:
                            metadata = await self._store_chunk(
                                chunk, doc_id, total_chunks)
                            logger.info(f"Stored chunk with ID: {chunk.get('id', 'generated')}")
                            logger.info(f"Metadata: {metadata}")
                        except Exception as e:
                            logger.error(f"Error storing chunk: {str(e)}")
            else:
                # For full documents, store without embeddings
                await self.client.get_or_create_collection(
                    name=f"{self.collection.name}_full",
                    metadata={"type": "full_documents"}
                ).add(
                    documents=[doc['content']],
                    ids=[str(uuid.uuid4())]
                )


async def embed_doc(cfg: DictConfig, chunked_docs: List[Dict]) -> None:
    logger.info("Starting document embedding process...")
    embedder = await Embedder.create_chromadb_client(
        cfg, SETTINGS.CHROMA_HOST, SETTINGS.CHROMA_PORT
    )
    embedder.embedding_fn = embedder._create_embedding_function(
        provider=cfg.llm.provider,
        model_name=cfg.llm.embedding_model,
        api_key=SETTINGS.OPENAI_API_KEY
    )
    embedder.collection = await embedder._get_or_create_collection(
        collection_name=cfg.embedder.collection,
        similarity_metric=cfg.embedder.similarity_metric
    )
    logger.info(f"Created Collection: {embedder.collection.name}")

    total_chunks = sum(
        len(doc['chunks']) if doc['type'] == 'chunked' else 1
        for doc in chunked_docs
    )
    logger.info(f"Processing {len(chunked_docs)} documents "
                f"with total {total_chunks} chunks")
    try:
        await embedder._store_processed_documents(chunked_docs)
        
        # Verify embeddings by checking collection count
        collection_count = await embedder.collection.count()
        logger.info(f"Successfully stored {collection_count}"
                    f"embeddings in collection")
        
        # Optional: Check a few random embeddings
        if collection_count > 0:
            sample = await embedder.collection.get(limit=1)
            if sample and 'embeddings' in sample:
                logger.info("Sample embedding verification successful")
            else:
                logger.warning("Sample embedding verification failed")              
    except Exception as e:
        logger.error(f"Error during document embedding: {str(e)}")
        raise e
    logger.info("Document embedding process completed")
    return None