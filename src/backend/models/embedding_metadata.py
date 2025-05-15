from typing import List
from pydantic import BaseModel


class EmbeddingMetadata(BaseModel):
    category: str    # high-level topic of content
    keywords: List[str]
    related_topics: List[str]