import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents according to the strategy."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of the strategy."""
        pass


class RecursiveCharacterChunkingStrategy(ChunkingStrategy):
    """Character-based chunking strategy: RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size, chunk_overlap)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n", "\n",
                "。", "！", "？",  # Chinese
                ". ", "! ", "? ",  # English
                "；", "：", "，", "、",  # Chinese
                "; ", ": ", ", ",  # English
                "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002",  # Special characters
                " ", ""
            ],
            keep_separator=True
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self._text_splitter.split_documents(documents)
        
        # Add strategy-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_type'] = 'recursive'
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    @property
    def strategy_name(self) -> str:
        return "recursive"


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic-based chunking strategy using embeddings."""
    
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        breakpoint_threshold_type: str,
        embedding_model: str = "text-embedding-3-small",
        buffer_size: int = 1,
        breakpoint_threshold_amount: float = 95.0,
        min_chunk_size: Optional[int] = None,
    ):
        super().__init__(chunk_size, chunk_overlap,)
        self.embedding_model = embedding_model
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"Initializing semantic chunking with embedding model: "
                    f"{embedding_model}")

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self._text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            buffer_size=self.buffer_size,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            min_chunk_size=self.min_chunk_size,
        )
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        try:
            chunks = self._text_splitter.split_documents(documents)
            logger.info("Using semantic chunker for document splitting")
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {str(e)}. "
                           "Falling back to recursive character splitting.")
            chunks = self._fallback_splitter.split_documents(documents)
        
        # Add strategy-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_type'] = 'semantic'
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['embedding_model'] = self.embedding_model
        
        return chunks
    
    @property
    def strategy_name(self) -> str:
        return "semantic"


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    
    @staticmethod
    def create_strategy(strategy: str, cfg: dict) -> ChunkingStrategy:
        """Create a chunking strategy based on specified type and configuration
        
        Args:
            strategy: The type of chunking strategy.
                ('recursive', 'semantic')
            cfg: Configuration dictionary with parameters
            
        Returns:
            An instance of a ChunkingStrategy
        """
        logger.info(f"Creating chunking strategy: {strategy}")
    
        if strategy == "recursive":
            return RecursiveCharacterChunkingStrategy(
                cfg['chunk_size'],
                cfg['chunk_overlap']
            )
        
        elif strategy == "semantic":
            return SemanticChunkingStrategy(
                cfg['chunk_size'],
                cfg['chunk_overlap'],
                cfg['breakpoint_threshold_type'],
                cfg['embedding_model'],
                cfg['buffer_size'],
                cfg['breakpoint_threshold_amount'],
                cfg['min_chunk_size']
            )
        
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}. "
                           "Using recursive as fallback.")
            return RecursiveCharacterChunkingStrategy(
                cfg['chunk_size'],
                cfg['chunk_overlap']
            )