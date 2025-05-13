from pydantic import BaseModel
from typing import Optional, List


class Course(BaseModel):
    course_id: str
    course_name: str
    course_level: str
    min_age: int
    max_age: int
    teacher: str
    start_date: str
    course_date_time: str
    student_count: int
    half_year_full_price: Optional[float]
    whole_year_full_price: Optional[float]
    half_year_discount_price: Optional[float]
    whole_year_discount_price: Optional[float]


class CourseFilter(BaseModel):
    age: int
    interests: Optional[List[str]] = None
