from uuid import UUID
import logging
from typing import List, Optional, Dict, Any

from core.db.repositories.parsed_document import ParsedDocumentRepository
from core.models.parsed_document import ParsedDocument, ParsedDocumentCreate, ParsedDocumentUpdate

# Configure logging
logger = logging.getLogger(__name__)

class ParsedDocumentService:
    """Service for parsed document operations."""
    
    def __init__(self):
        self.repository = ParsedDocumentRepository()
    
    async def create_parsed_document(self, parsed_document: ParsedDocumentCreate) -> ParsedDocument:
        """Create a new parsed document."""
        try:
            return await self.repository.create(parsed_document)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise ValueError(f"Database connection error: {str(e)}")
    
    async def get_parsed_document(self, parsed_document_id: UUID) -> Optional[ParsedDocument]:
        """Get a parsed document by ID."""
        try:
            return await self.repository.get_by_id(parsed_document_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    async def list_parsed_documents_by_user(self, user_id: UUID) -> List[ParsedDocument]:
        """List all parsed documents for a user."""
        try:
            return await self.repository.list_by_user(user_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return []
    
    async def list_parsed_documents_by_type(self, document_type_id: UUID) -> List[ParsedDocument]:
        """List all parsed documents for a document type."""
        try:
            return await self.repository.list_by_document_type(document_type_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return []
    
    async def update_parsed_document(self, parsed_document_id: UUID, parsed_document: ParsedDocumentUpdate) -> Optional[ParsedDocument]:
        """Update an existing parsed document."""
        try:
            # Check if document exists
            existing = await self.repository.get_by_id(parsed_document_id)
            if not existing:
                return None
            
            return await self.repository.update(parsed_document_id, parsed_document)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    async def delete_parsed_document(self, parsed_document_id: UUID) -> bool:
        """Delete a parsed document."""
        try:
            # Check if document exists
            existing = await self.repository.get_by_id(parsed_document_id)
            if not existing:
                return False
            
            return await self.repository.delete(parsed_document_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return False
    
    async def get_by_document_type_and_filename(self, document_type_id: UUID, user_id: UUID, original_file_name: str) -> Optional[ParsedDocument]:
        """Get a parsed document by document type ID, user ID and original filename."""
        try:
            return await self.repository.get_by_document_type_and_filename(document_type_id, user_id, original_file_name)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return None

    async def delete_by_document_type(self, document_type_id: UUID) -> bool:
        """Delete all parsed documents for a specific document type."""
        try:
            return await self.repository.delete_by_document_type(document_type_id)
        except ConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            return False