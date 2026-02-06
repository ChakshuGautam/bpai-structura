import json
import logging
from uuid import UUID
from typing import List, Optional, Dict, Any
from postgrest.exceptions import APIError

from core.db.client import supabase
from core.models.document_type import DocumentType, DocumentTypeCreate, DocumentTypeUpdate

# Configure logging
logger = logging.getLogger(__name__)

class DocumentTypeRepository:
    """Repository for document type CRUD operations using Supabase."""
    
    TABLE_NAME = "document_types"
    
    def _check_db_connection(self):
        """Check if the database connection is available."""
        if not supabase:
            logger.warning("Supabase client not initialized. Database operations will fail.")
            raise ConnectionError("Database not configured. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    async def create(self, document_type: DocumentTypeCreate) -> DocumentType:
        """Create a new document type in the database."""
        self._check_db_connection()
        
        try:
            data = document_type.model_dump()
            
            # Convert UUID to string if present
            if 'id' in data and data['id']:
                data['id'] = str(data['id'])
            
            # Convert user_id to string
            if 'user_id' in data and data['user_id']:
                data['user_id'] = str(data['user_id'])
            
            # Insert data using Supabase client
            result = supabase.table(self.TABLE_NAME).insert(data).execute()
            
            if not result.data or len(result.data) == 0:
                raise ValueError(f"Failed to create document type")
            
            return DocumentType.model_validate(result.data[0])
            
        except APIError as e:
            if "unique constraint" in str(e).lower() and "document_types_user_id_type_key" in str(e):
                raise ValueError(f"A document type with the same name already exists for this user")
            raise ValueError(f"Database error: {str(e)}")
    
    async def get_by_id(self, document_type_id: UUID) -> Optional[DocumentType]:
        """Get a document type by its ID."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME).select("*").eq("id", str(document_type_id)).execute()
        
        if not result.data or len(result.data) == 0:
            return None
        
        return DocumentType.model_validate(result.data[0])
    
    async def get_by_type_and_user(self, type_name: str, user_id: UUID) -> Optional[DocumentType]:
        """Get a document type by its type name and user_id."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME)\
            .select("*")\
            .eq("type", type_name)\
            .eq("user_id", str(user_id))\
            .execute()
        
        if not result.data or len(result.data) == 0:
            return None
        
        return DocumentType.model_validate(result.data[0])
    
    async def list_by_user(self, user_id: UUID) -> List[DocumentType]:
        """Get all document types for a specific user."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME)\
            .select("*")\
            .eq("user_id", str(user_id))\
            .execute()
        
        if not result.data:
            return []
        
        return [DocumentType.model_validate(row) for row in result.data]
    
    async def list_all(self) -> List[DocumentType]:
        """Get all document types."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME).select("*").execute()
        
        if not result.data:
            return []
        
        return [DocumentType.model_validate(row) for row in result.data]
    
    async def update(self, document_type_id: UUID, document_type: DocumentTypeUpdate) -> Optional[DocumentType]:
        """Update an existing document type."""
        self._check_db_connection()
        
        try:
            # Only include non-None fields in the update
            update_data = {k: v for k, v in document_type.model_dump().items() if v is not None}
            
            if not update_data:
                # Nothing to update
                return await self.get_by_id(document_type_id)
            
            # Update data using Supabase client
            result = supabase.table(self.TABLE_NAME).update(update_data).eq("id", str(document_type_id)).execute()
            
            if not result.data or len(result.data) == 0:
                return None
            
            return DocumentType.model_validate(result.data[0])
            
        except APIError as e:
            if "unique constraint" in str(e).lower() and "document_types_user_id_type_key" in str(e):
                raise ValueError(f"Cannot update: another document type with the same name exists for this user")
            raise ValueError(f"Database error: {str(e)}")
    
    async def delete(self, document_type_id: UUID) -> bool:
        """Delete a document type."""
        self._check_db_connection()
        
        result = supabase.table(self.TABLE_NAME).delete().eq("id", str(document_type_id)).execute()
        
        # Return True if something was deleted, False otherwise
        return bool(result.data and len(result.data) > 0)
