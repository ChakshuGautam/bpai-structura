from typing import Generic, TypeVar, Optional, List, Any, Dict
from pydantic import BaseModel

# Generic type for response data
T = TypeVar('T')


class BaseResponse(BaseModel):
    """Base response model with success flag."""
    success: bool


class ErrorResponse(BaseResponse):
    """Error response model with error details."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseResponse, Generic[T]):
    """Success response model with data."""
    success: bool = True
    data: T


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated response model with data and pagination info."""
    success: bool = True
    data: List[T]
    total: int
    page: int
    page_size: int
    pages: int
