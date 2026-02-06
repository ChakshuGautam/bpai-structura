import logging
from uuid import UUID
from typing import List, Optional, Dict, Any
from postgrest.exceptions import APIError

from core.db.client import supabase
from core.models.parsed_document import ParsedDocument, ParsedDocumentCreate, ParsedDocumentUpdate

# Configure logging
logger = logging.getLogger(__name__)

class ParsedDocumentRepository:
    """Repository for parsed document CRUD operations using Supabase."""
    
    TABLE_NAME = "parsed_documents"
    
    def _check_db_connection(self):
        """Check if the database connection is available."""
        if not supabase:
            logger.warning("Supabase client not initialized. Database operations will fail.")
            raise ConnectionError("Database not configured. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    async def create(self, parsed_document: ParsedDocumentCreate) -> ParsedDocument:
        """Create a new parsed document entry in the database."""
        self._check_db_connection()
        
        try:
            data = parsed_document.model_dump()
            
            # Convert UUIDs to strings
            if 'id' in data and data['id']:
                data['id'] = str(data['id'])
            if 'user_id' in data and data['user_id']:
                data['user_id'] = str(data['user_id'])
            if 'document_type_id' in data and data['document_type_id']:
                data['document_type_id'] = str(data['document_type_id'])
            if 'file_id' in data and data['file_id']:
                data['file_id'] = str(data['file_id'])
            
            # Insert data using Supabase client
            result = supabase.table(self.TABLE_NAME).insert(data).execute()
            
            if not result.data or len(result.data) == 0:
                raise ValueError(f"Failed to create parsed document entry")
            
            return ParsedDocument.model_validate(result.data[0])
            
        except APIError as e:
            raise ValueError(f"Database error: {str(e)}")
    
    async def get_by_id(self, parsed_document_id: UUID) -> Optional[ParsedDocument]:
        """Get a parsed document by its ID."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME).select("*").eq("id", str(parsed_document_id)).execute()
        
        if not result.data or len(result.data) == 0:
            return None
        
        return ParsedDocument.model_validate(result.data[0])
    
    async def list_by_user(self, user_id: UUID) -> List[ParsedDocument]:
        """Get all parsed documents for a specific user."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME)\
            .select("*")\
            .eq("user_id", str(user_id))\
            .execute()
        
        if not result.data:
            return []
        
        return [ParsedDocument.model_validate(row) for row in result.data]
    
    async def list_by_document_type(self, document_type_id: UUID) -> List[ParsedDocument]:
        """Get all parsed documents for a specific document type."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME)\
            .select("*")\
            .eq("document_type_id", str(document_type_id))\
            .execute()
        
        if not result.data:
            return []
        
        return [ParsedDocument.model_validate(row) for row in result.data]
    
    async def update(self, parsed_document_id: UUID, parsed_document: ParsedDocumentUpdate) -> Optional[ParsedDocument]:
        """Update an existing parsed document."""
        self._check_db_connection()
        
        try:
            # Only include non-None fields in the update
            update_data = {k: v for k, v in parsed_document.model_dump().items() if v is not None}
            
            if not update_data:
                # Nothing to update
                return await self.get_by_id(parsed_document_id)
            
            # Add updated_at timestamp
            update_data['updated_at'] = 'NOW()'
            
            # Update data using Supabase client
            result = supabase.table(self.TABLE_NAME).update(update_data).eq("id", str(parsed_document_id)).execute()
            
            if not result.data or len(result.data) == 0:
                return None
            
            return ParsedDocument.model_validate(result.data[0])
            
        except APIError as e:
            raise ValueError(f"Database error: {str(e)}")
    
    async def delete(self, parsed_document_id: UUID) -> bool:
        """Delete a parsed document."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME).delete().eq("id", str(parsed_document_id)).execute()
        
        # Return True if something was deleted, False otherwise
        return bool(result.data and len(result.data) > 0)
    
    async def get_by_document_type_and_filename(self, document_type_id: UUID, user_id: UUID, original_file_name: str) -> Optional[ParsedDocument]:
        """Get a parsed document by document type ID, user ID and original filename."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME)\
            .select("*")\
            .eq("document_type_id", str(document_type_id))\
            .eq("user_id", str(user_id))\
            .eq("original_file_name", original_file_name)\
            .execute()
        
        if not result.data or len(result.data) == 0:
            return None
        
        return ParsedDocument.model_validate(result.data[0])

    async def delete_by_document_type(self, document_type_id: UUID) -> bool:
        """Delete all parsed documents for a specific document type."""
        self._check_db_connection()
        
        try:
            result = supabase.table(self.TABLE_NAME).delete().eq("document_type_id", str(document_type_id)).execute()
            
            # Return True if something was deleted, False otherwise
            return bool(result.data and len(result.data) > 0)
        except APIError as e:
            logger.error(f"Database error deleting parsed documents by document type: {str(e)}")
            raise ValueError(f"Database error: {str(e)}")