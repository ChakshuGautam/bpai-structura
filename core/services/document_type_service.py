from uuid import UUID
import logging
from typing import List, Optional, Dict, Any

from core.db.repositories.document_type import DocumentTypeRepository
from core.models.document_type import DocumentType, DocumentTypeCreate, DocumentTypeUpdate

# Configure logging
logger = logging.getLogger(__name__)

class DocumentTypeService:
    """Service for document type operations."""
    
    def __init__(self):
        self.repository = DocumentTypeRepository()
    
    async def create_document_type(self, document_type: DocumentTypeCreate) -> DocumentType:
        """Create a new document type."""
        try:
            # Check if document type with the same type already exists for this user
            existing = await self.repository.get_by_type_and_user(document_type.type, document_type.user_id)
            if existing:
                raise ValueError(f"Document type with type '{document_type.type}' already exists for this user")
            
            return await self.repository.create(document_type)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise ValueError(f"Database connection error: {str(e)}")
    
    async def get_document_type(self, document_type_id: UUID) -> Optional[DocumentType]:
        """Get a document type by ID."""
        try:
            return await self.repository.get_by_id(document_type_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    async def get_document_type_by_type(self, type_name: str, user_id: UUID) -> Optional[DocumentType]:
        """Get a document type by its type name and user ID."""
        try:
            return await self.repository.get_by_type_and_user(type_name, user_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    async def list_document_types(self) -> List[DocumentType]:
        """List all document types."""
        try:
            return await self.repository.list_all()
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return []
    
    async def update_document_type(self, document_type_id: UUID, document_type: DocumentTypeUpdate) -> Optional[DocumentType]:
        """Update an existing document type."""
        try:
            # Check if document type exists
            existing = await self.repository.get_by_id(document_type_id)
            if not existing:
                return None
            
            # If type is being changed, check if new type already exists for this user
            if document_type.type and document_type.type != existing.type:
                type_exists = await self.repository.get_by_type_and_user(document_type.type, existing.user_id)
                if type_exists:
                    raise ValueError(f"Document type with type '{document_type.type}' already exists for this user")
            
            return await self.repository.update(document_type_id, document_type)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    async def delete_document_type(self, document_type_id: UUID) -> bool:
        """Delete a document type."""
        try:
            # Check if document type exists
            existing = await self.repository.get_by_id(document_type_id)
            if not existing:
                return False
            
            return await self.repository.delete(document_type_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return False
    
    async def validate_document_against_type(self, document_data: Dict[str, Any], document_type_id: UUID) -> Dict[str, Any]:
        """
        Validate a document against a document type's schema.
        This is a placeholder for future implementation with JSONSchema validation.
        """
        try:
            document_type = await self.repository.get_by_id(document_type_id)
            if not document_type:
                raise ValueError(f"Document type with ID '{document_type_id}' not found")
            
            # TODO: Implement proper JSONSchema validation using jsonschema library
            # For now, just return the document data
            return document_data
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            # If database is not available, just return the document data as-is
            return document_data
    
    async def get_by_type_and_user(self, type_name: str, user_id: UUID) -> Optional[DocumentType]:
        """Get a document type by its type name and user_id."""
        return await self.repository.get_by_type_and_user(type_name, user_id)
    
    async def list_document_types_by_user(self, user_id: UUID) -> List[DocumentType]:
        """List all document types for a specific user."""
        return await self.repository.list_by_user(user_id)
