import logging
from typing import List, Dict, Union
import pandas as pd
import tiktoken
from omegaconf import DictConfig
from langchain.schema import Document
from src.backend.dataloaders.local_doc_loader import (
    LoadedUnstructuredDocument,
    LoadedStructuredDocument
)
from src.backend.dataprocessor.chunker_factory import ChunkingStrategyFactory

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(
        self,
        token_threshold: int,
        chunking_config: Dict
    ):
        self.token_threshold = token_threshold
        self.chunking_config = chunking_config
        self.encoders = {}  # Cache for encoders
        self.chunking_strategy = chunking_config.get('strategy', 'recursive')
        self.chunking_strategy = ChunkingStrategyFactory.create_strategy(
            self.chunking_strategy,
            chunking_config
        )
    
    def _tokenizer(self, model: str) -> tiktoken.Encoding:
        """Get or create a token encoder for the specified model."""
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]

    def _get_token_count(self, text: str, model: str) -> int:
        """Count tokens in a text using tiktoken."""
        if not isinstance(text, str):
            text = str(text)
        
        # Clean the text
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = ' '.join(text.split())   # Normalize whitespace
        
        try:
            encoder = self._tokenizer(model)
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def _chunk_structured_doc(self, doc: pd.DataFrame, metadata: dict = None):
        """Process structured data (DataFrame) by chunking rows."""
        if metadata is None:
            metadata = {}
        logger.info(f"Metadata before chunking Structured data: {metadata}")
        rows_threshold = metadata.get('rows_threshold', 50)
        total_rows = len(doc)
        logger.info(f"Total rows in DataFrame: {total_rows}")

        if total_rows <= rows_threshold:
            return {
                'type': 'full',
                'content': doc.to_dict(orient='records'),
                'metadata': metadata
            }

        chunks = []
        for index, row in doc.iterrows():
            row_text = " ".join([
                f"{col}: {str(val)}"
                for col, val in row.items()
            ])
            chunk_metadata = {
                **metadata,
                'is_structured': True,
                'chunk_type': 'row',
                'chunk_index': index,
                'total_chunks': total_rows
            }
            chunks.append({
                'content': row_text,
                'metadata': chunk_metadata
            })
        logger.info(f"Structured data: {len(chunks)} chunks")
        return {
            'type': 'chunked',
            'chunks': chunks,
            'num_chunks': len(chunks)
        }
    
    def _chunk_unstructured_doc(
        self, doc: str, model: str, metadata: dict = None
    ):
        """Process unstructured text data based on token count."""
        if not isinstance(doc, str):
            doc = str(doc)
        
        # Clean the text before processing
        text = doc.replace('\n', ' ')  # Replace newlines with spaces
        text = ' '.join(text.split())   # Normalize whitespace

        if metadata is None:
            metadata = {}
        logger.info(f"Metadata before chunking Unstructured data: {metadata}")

        token_count = self._get_token_count(text, model)
        total_characters = len(text)

        if token_count <= self.token_threshold:
            # Small documents: store as-is for direct LLM search
            full_text_info = {
                'type': 'full',
                'content': text,
                'metadata': metadata,
                'total_tokens': token_count,
                'total_characters': total_characters
            }
            logger.info(f"Document stored as full text, {token_count} tokens")
            logger.info(f"Total characters: {total_characters}")
            return full_text_info
        else:
            # Large documents: chunk using LangChain's splitter
            # Create a LangChain Document to use with the splitter
            doc = Document(page_content=text, metadata=metadata)
            chunks = self.chunking_strategy.split_documents([doc])
            logger.info(f"Unstructured data: {len(chunks)} chunks using "
                        f"{self.chunking_strategy.strategy_name} strategy")
            for chunk in chunks:
                chunk.metadata.update({
                    'is_structured': False,
                    'chunk_type': f"{self.chunking_strategy}",
                    'chunk_index': chunk.metadata.get('chunk_index', 0),
                    'total_chunks': len(chunks)
                })

            # Log first chunk details
            if chunks:
                first_chunk = chunks[0].page_content
                first_chunk_chars = len(first_chunk)
                first_chunk_tokens = self._get_token_count(first_chunk, model)
                logger.info(f"First chunk: {first_chunk_tokens} tokens, "
                            f"{first_chunk_chars} characters")

            return {
                'type': 'chunked',
                'chunks': [{
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                } for chunk in chunks],
                'num_chunks': len(chunks),
                'chunking_strategy': self.chunking_strategy.strategy_name
            }

    def _chunk_single_doc(
        self,
        doc: Union[str, pd.DataFrame],
        model: str,
        metadata: Dict = None
    ) -> Dict:
        """Process a document based on its type and token count."""
        if isinstance(doc, pd.DataFrame):
            return self._chunk_structured_doc(doc, metadata)
        else:
            return self._chunk_unstructured_doc(doc, model, metadata)


def batch_chunk_doc(
    cfg: DictConfig,
    documents: List[Union[LoadedUnstructuredDocument, LoadedStructuredDocument]]
) -> List[Dict]:
    """Entry point for chunking documents.
    Args:
        cfg (DictConfig): Configuration object.
        documents (List[Union[LoadedUnstructuredDocument,
            LoadedStructuredDocument]]):
            List of documents to chunk.
            
    Returns:
        List[Dict]: List of chunked documents.
    """
    chunking_config = {
        'strategy': cfg.chunker.get('strategy', 'recursive'),
        'chunk_size': cfg.chunker.recursive.chunk_size,
        'chunk_overlap': cfg.chunker.recursive.chunk_overlap,
        'embedding_model': cfg.chunker.get(
            'embedding_model', 'text-embedding-3-small'
        ),
        'buffer_size': cfg.chunker.semantic.get('buffer_size', 1),
        'breakpoint_threshold_type': cfg.chunker.semantic.get(
            'breakpoint_threshold_type', 'percentile'
        ),
        'breakpoint_threshold_amount': cfg.chunker.semantic.get(
            'breakpoint_threshold_amount', 95.0
        ),
        'min_chunk_size': cfg.chunker.semantic.get('min_chunk_size', None),
    }
    chunker = Chunker(
        token_threshold=cfg.chunker.token_threshold,
        chunking_config=chunking_config
    )

    chunked_doc = []
    detailed_chunk_info = []
    for i, doc in enumerate(documents):
        if hasattr(doc, 'content') and hasattr(doc, 'metadata'):
            content = doc.content
            metadata = doc.metadata
        elif isinstance(doc, dict):
            content = doc['content']
            metadata = doc.get('metadata', {})
        else:
            content = doc
            metadata = {}

        chunked_result = chunker._chunk_single_doc(
            content, cfg.llm.model, metadata
        )
        chunked_doc.append(chunked_result)
        
        # Collect detailed chunk information
        detailed_chunk_info.append({
            'document_index': i,
            'total_chunks': len(chunked_result['chunks']),
            # 'chunk_contents': [chunk['content'][:100] + '...' for chunk in chunked_result['chunks']]
        })
    
    # Log detailed chunk information
    logger.info("Detailed Chunk Information:")
    for info in detailed_chunk_info:
        logger.info(f"Document {info['document_index']}: "
                    f"{info['total_chunks']} chunks")
    
    total_chunks = sum(len(doc['chunks']) for doc in chunked_doc)
    logger.info(f"Total {total_chunks} chunks.")
    
    return chunked_doc