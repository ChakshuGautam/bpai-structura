from uuid import UUID, uuid4
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DocumentTypeBase(BaseModel):
    """Base model for document types with common fields."""
    user_id: UUID = Field(..., description="ID of the user who owns this document type")
    type: str = Field(..., description="Document type name (e.g., 'Invoice', 'Contract')")
    description: str = Field(..., description="Human-readable description of the document type")
    schema: Dict[str, Any] = Field(..., description="JSONSchema defining the expected fields and their types")


class DocumentTypeCreate(DocumentTypeBase):
    """Model for creating a new document type."""
    pass


class DocumentTypeUpdate(BaseModel):
    """Model for updating an existing document type."""
    type: Optional[str] = Field(None, description="Document type name (e.g., 'Invoice', 'Contract')")
    description: Optional[str] = Field(None, description="Human-readable description of the document type")
    schema: Optional[Dict[str, Any]] = Field(None, description="JSONSchema defining the expected fields and their types")

    @field_validator('schema')
    def validate_schema(cls, v):
        """Validate that the schema is a valid JSONSchema."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Schema must be a valid JSON object")
        return v


class DocumentType(DocumentTypeBase):
    """Complete document type model that includes the ID."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the document type")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True
