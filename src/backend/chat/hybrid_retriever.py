from typing import List, Dict, Any
import json
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch
from utils.settings import SETTINGS


class SearchMetadata(BaseModel):
    category: str
    keywords: List[str]
    related_topics: List[str]


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: SearchMetadata


class HybridRetriever:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(
            path=self.cfg.hybrid_retriever.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=SETTINGS.OPENAI_API_KEY,
            model_name=self.cfg.llm.embedding_model
        )
        self.collection = self.client.get_collection(
            name=self.cfg.hybrid_retriever.collection,
            embedding_function=self.embedding_function
        )
        if cfg.hybrid_retriever.use_reranker:
            self.reranker = CrossEncoder(cfg.hybrid_retriever.reranker_model)
        else:
            self.reranker = None
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization of scores"""
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _get_keyword_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Get BM25 scores for keyword matching
        
        BM25 calculates a score based on Term Frequency 
        (how often words appear in the document), normalized longer documents,
        inverses document frequency (how common words are across documents). 
        Similar to TD-IDF + length normilization, etc. 
        """
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        # Create BM25 index from tokenized documents
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get keyword scores
        tokenized_query = query.lower().split()
        # Get BM25 scores for how well the query matches each document
        keyword_scores = bm25.get_scores(tokenized_query)
        # 0-1 normalize scores
        return self._normalize_scores(keyword_scores.tolist())

    async def _rerank_results(
        self, query: str, search_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank search results using cross-encoder model"""
        if not self.reranker or not search_results:
            return search_results
        # Prepare query-document pairs for reranking
        query_doc_pairs = [(query, result.content) for result in search_results]
        # Get scores from cross-encoder
        with torch.no_grad():
            scores = self.reranker.predict(query_doc_pairs)
        # Update scores in search results
        for i, score in enumerate(scores):
            search_results[i].score = float(score)
        # Return reranked results
        return sorted(search_results, key=lambda x: x.score, reverse=True)

    async def search(
        self,
        query: str,
        filter_conditions: Dict = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filter_conditions: Optional filters for metadata fields
        """
        # Get semantic search results with scores
        results = self.collection.query(
            query_texts=[query],
            n_results=self.cfg.hybrid_retriever.top_k,
            where=filter_conditions,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Accessing the inner lists of results. Chromadb supports batch queries
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        # Convert distances to similarity scores (1 - distance)
        semantic_scores = self._normalize_scores(
            [1 - d for d in results['distances'][0]]
        )
        
        # Get keyword search scores
        keyword_scores = self._get_keyword_scores(query, documents)
        
        # Combine scores
        combined_scores = [
            (self.cfg.hybrid_retriever.semantic_weight * ss + self.cfg.hybrid_retriever.keyword_weight * ks)
            for ss, ks in zip(semantic_scores, keyword_scores)
        ]
        
        # Sort results by combined score
        search_results = []
        for doc, meta, score in zip(documents, metadatas, combined_scores):
            keywords = json.loads(meta.get('keywords', '[]'))
            topics = json.loads(meta.get('related_topics', '[]'))
            metadata_object = SearchMetadata(
                category=meta.get('category', ''),
                keywords=keywords,
                related_topics=topics
            )

            search_results.append(
                SearchResult(
                    content=doc,
                    score=score,
                    metadata=metadata_object
                )
            )
        search_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        if self.cfg.hybrid_retriever.use_reranker:
            search_results = await self._rerank_results(query, search_results)
        search_results = search_results[:self.cfg.hybrid_retriever.reranker_top_k]
        return search_results    
        