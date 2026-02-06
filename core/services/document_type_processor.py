import jsonschema
from typing import Dict, Any, Optional
from uuid import UUID

from core.services.document_type_service import DocumentTypeService


class DocumentTypeProcessor:
    """
    Processes documents based on their document type.
    This class will validate and structure document data according to the document type's schema.
    """
    
    def __init__(self):
        self.document_type_service = DocumentTypeService()
    
    async def process_document(self, document_data: Dict[str, Any], document_type_id: UUID) -> Dict[str, Any]:
        """
        Process a document according to its document type.
        
        Args:
            document_data: The document data to process
            document_type_id: The ID of the document type to use for processing
            
        Returns:
            The processed document data
        """
        # Get the document type
        document_type = await self.document_type_service.get_document_type(document_type_id)
        if not document_type:
            raise ValueError(f"Document type with ID '{document_type_id}' not found")
        
        # Validate the document data against the schema
        try:
            jsonschema.validate(instance=document_data, schema=document_type.schema)
        except jsonschema.exceptions.ValidationError as e:
            # Log the validation error
            print(f"Document validation error: {str(e)}")
            # Return the original data with validation errors
            return {
                "original_data": document_data,
                "validation_errors": str(e),
                "valid": False
            }
        
        # For now, just return the validated document data
        # In a more complete implementation, this would apply additional processing
        # based on the document type
        return {
            "processed_data": document_data,
            "document_type": document_type.type,
            "valid": True
        }
    
    async def detect_document_type(self, document_data: Dict[str, Any]) -> Optional[UUID]:
        """
        Detect the document type based on the document data.
        This is a placeholder for a more sophisticated implementation.
        
        Args:
            document_data: The document data to analyze
            
        Returns:
            The ID of the detected document type, or None if no match is found
        """
        # Get all document types
        document_types = await self.document_type_service.list_document_types()
        
        # Try to match against each document type's schema
        for document_type in document_types:
            try:
                jsonschema.validate(instance=document_data, schema=document_type.schema)
                # If validation succeeds, return this document type's ID
                return document_type.id
            except jsonschema.exceptions.ValidationError:
                # Validation failed, try the next document type
                continue
        
        # No matching document type found
        return None 