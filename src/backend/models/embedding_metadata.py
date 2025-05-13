from typing import List
from pydantic import BaseModel


class EmbeddingMetadata(BaseModel):
    category: str
    keywords: List[str]
    related_topics: List[str]