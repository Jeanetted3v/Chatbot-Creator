from typing import Any
from bson import ObjectId


def serialize_mongodb_doc(obj: Any) -> Any:
    """Recursively converts ObjectId to string in any data structure"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_mongodb_doc(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_mongodb_doc(item) for item in obj]
    return obj