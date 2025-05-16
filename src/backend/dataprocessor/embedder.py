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

    async def _create_collection(
        self,
        collection_name: str,
        similarity_metric: str,
        metadata: Optional[Dict] = None,
    ):
        if metadata is None:
            metadata = {"hnsw:space": similarity_metric}
            
        return await self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata
        )
    
    async def _extract_metadata(self, content: str) -> EmbeddingMetadata:
        logger.info("Start Extracting metadata")
        result = await self.agent.run(
            self.prompts['user_prompt'].format(content=content)
        )
        metadata = result.data
        logger.info(f"Extracted metadata: {metadata}")
        return metadata

    def _convert_metadata_str(self, metadata: Dict) -> Dict:
        return {
            key: json.dumps(value) if not isinstance(
                value,
                (str, int, float, bool)
            )
            else value
            for key, value in metadata.items()
        }

    async def _store_processed_documents(self, processed_docs: List[Dict]):
        """
        Store processed documents in ChromaDB.
        Handles both chunked and full documents appropriately.
        """
        for doc in processed_docs:
            if doc['type'] == 'chunked':
                chunk_metadatas = []
                chunk_embeddings = []
                chunk_documents = []
                chunk_ids = []
                # For chunked documents, store each chunk with its embedding
                # Use add method, upsert might overwrite existing embeddings
                for chunk in doc['chunks']:
                    extracted_metadata = await self._extract_metadata(
                        chunk['content']
                    )
                    enhanced_metadata = {
                        **chunk['metadata'],
                        **extracted_metadata.model_dump(),
                        'chunk_type': 'partial',
                        'total_chunks': doc['num_chunks'],
                        'doc_id': doc.get('doc_id', str(uuid.uuid4()))
                    }
                    metadata = self._convert_metadata_str(enhanced_metadata)
                    embedding = self.embedding_fn(chunk['content'])
                    if isinstance(embedding, list) and len(embedding) == 1:
                        embedding = embedding[0]  # Extract the numpy array from the list
                    chunk_embeddings.append(embedding)
                    chunk_documents.append(chunk['content'])
                    chunk_metadatas.append(metadata)
                    chunk_ids.append(str(uuid.uuid4()))
                    logger.info(f"Storing chunk with metadata: {metadata}")
                await self.collection.add(
                    documents=chunk_documents,
                    embeddings=chunk_embeddings,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
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
    embedding_fn = embedder._create_embedding_function(
        provider=cfg.llm.provider,
        model_name=cfg.llm.embedding_model,
        api_key=SETTINGS.OPENAI_API_KEY
    )
    embedder.embedding_fn = embedding_fn
    embedder.collection = await embedder._create_collection(
        collection_name=cfg.embedder.collection,
        similarity_metric=cfg.embedder.similarity_metric,
        # embedding_function=embedding_fn
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