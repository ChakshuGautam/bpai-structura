from uuid import UUID, uuid4
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class ParsedDocumentBase(BaseModel):
    """Base model for parsed document data."""
    user_id: UUID = Field(..., description="ID of the user who owns this document")
    document_type_id: UUID = Field(..., description="ID of the document type used for parsing")
    data: Dict[str, Any] = Field(..., description="Parsed data from the document")
    original_file_name: str = Field(..., description="Original file name")
    original_file_url: Optional[str] = Field(None, description="URL to the original file if stored")
    file_id: Optional[UUID] = Field(None, description="ID of the file from user_files table, if available")

class ParsedDocumentCreate(ParsedDocumentBase):
    """Model for creating a new parsed document entry, including file_id if available."""
    pass

class ParsedDocumentUpdate(BaseModel):
    """Model for updating an existing parsed document."""
    data: Optional[Dict[str, Any]] = Field(None, description="Updated parsed data")
    original_file_url: Optional[str] = Field(None, description="Updated file URL")

class ParsedDocument(ParsedDocumentBase):
    """Complete parsed document model that includes the ID, timestamps, and file_id."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the parsed document")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True