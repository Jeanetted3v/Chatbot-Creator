from typing import List, Optional
from enum import Enum
from pydantic import BaseModel


class IntentType(str, Enum):
    COURSE_INQUIRY = "course_inquiry"
    SCHEDULE_INQUIRY = "schedule_inquiry"
    FEE_INQUIRY = "fee_inquiry"
    GENERAL_INQUIRY = "general_inquiry"


class QueryParameters(BaseModel):
    age: Optional[int]
    subject: Optional[str]
    english_level: Optional[str]
    lexile_score: Optional[str]
    original_query: str


class IntentResult(BaseModel):
    intent: IntentType
    parameters: QueryParameters
    response: Optional[str]
    missing_info: List[str]
