from uuid import UUID
from typing import List
import logging

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
import os
from tempfile import NamedTemporaryFile
import shutil

from core.models.document_type import DocumentType, DocumentTypeCreate, DocumentTypeUpdate
from core.models.responses import SuccessResponse, ErrorResponse
from core.services.document_type_service import DocumentTypeService
from core.api.dependencies import verify_api_key  # Import the common dependency
from core.hack.get_domain_schema import detect_document_schema, extract_document_data_with_schema
from core.models.parsed_document import ParsedDocumentCreate
from core.services.parsed_document_service import ParsedDocumentService
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/document-types", tags=["Document Types"])

# Dependency to get DocumentTypeService
async def get_document_type_service():
    """Get document type service dependency."""
    return DocumentTypeService()

# Add service dependency
async def get_parsed_document_service():
    """Get parsed document service dependency."""
    return ParsedDocumentService()


@router.post(
    "",
    response_model=SuccessResponse[DocumentType],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new document type",
    dependencies=[Depends(verify_api_key)]
)
async def create_document_type(
    document_type: DocumentTypeCreate,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    Create a new document type with the provided details.
    """
    try:
        # Check if document type already exists for this user
        existing_type = await service.get_by_type_and_user(document_type.type, document_type.user_id)
        if existing_type:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document type '{document_type.type}' already exists for this user"
            )
        
        created_type = await service.create_document_type(document_type)
        return SuccessResponse(data=created_type)
    except ValueError as e:
        if "Database connection error" in str(e):
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service unavailable. Please try again later."
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating document type: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document type: {str(e)}"
        )


@router.get(
    "",
    response_model=SuccessResponse[List[DocumentType]],
    summary="List document types for a user",
    dependencies=[Depends(verify_api_key)]
)
async def list_document_types(
    user_id: UUID,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    List all document types for a specific user.
    """
    try:
        document_types = await service.list_document_types_by_user(user_id)
        return SuccessResponse(data=document_types)
    except Exception as e:
        logger.error(f"Error listing document types: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list document types: {str(e)}"
        )


@router.get(
    "/{document_type_id}",
    response_model=SuccessResponse[DocumentType],
    summary="Get a document type by ID",
    dependencies=[Depends(verify_api_key)]
)
async def get_document_type(
    document_type_id: UUID,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    Get a document type by its unique identifier.
    """
    document_type = await service.get_document_type(document_type_id)
    if not document_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document type with ID {document_type_id} not found"
        )
    return SuccessResponse(data=document_type)


@router.put(
    "/{document_type_id}",
    response_model=SuccessResponse[DocumentType],
    summary="Update a document type",
    dependencies=[Depends(verify_api_key)]
)
async def update_document_type(
    document_type_id: UUID,
    document_type: DocumentTypeUpdate,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    Update an existing document type with the provided details.
    """
    try:
        # Get existing document type
        existing = await service.get_document_type(document_type_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document type with ID {document_type_id} not found"
            )
        
        # If type is being updated, check for conflicts
        if document_type.type and document_type.type != existing.type:
            conflict = await service.get_by_type_and_user(document_type.type, existing.user_id)
            if conflict:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Document type '{document_type.type}' already exists for this user"
                )
        
        updated_type = await service.update_document_type(document_type_id, document_type)
        return SuccessResponse(data=updated_type)
    except ValueError as e:
        if "Database connection error" in str(e):
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service unavailable. Please try again later."
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating document type: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document type: {str(e)}"
        )


@router.delete(
    "/{document_type_id}",
    response_model=SuccessResponse[bool],
    summary="Delete a document type",
    dependencies=[Depends(verify_api_key)]
)
async def delete_document_type(
    document_type_id: UUID,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    Delete a document type by its unique identifier.
    """
    result = await service.delete_document_type(document_type_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document type with ID {document_type_id} not found"
        )
    return SuccessResponse(data=True)


@router.post(
    "/infer-schema",
    response_model=SuccessResponse[dict],
    summary="Infer document schema from PDF and optionally save as DocumentType",
    dependencies=[Depends(verify_api_key)]
)
async def infer_document_schema(
    file: UploadFile = File(...),
    user_id: UUID = Form(...),
    type: str = Form(None),
    description: str = Form(None),
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    Upload a PDF and infer its schema. Optionally save as a DocumentType.
    If document type already exists for the user, return the existing schema.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    # If type is provided, first check if document type exists
    if type:
        existing_type = await service.get_by_type_and_user(type, user_id)
        if existing_type:
            # Return existing schema if found
            return SuccessResponse(data={
                "schema": existing_type.schema,
                "document_type_id": str(existing_type.id),
                "message": f"Using existing schema for document type '{type}'"
            })

    tmp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        # Save uploaded PDF to temp file
        with tmp_pdf as f:
            shutil.copyfileobj(file.file, f)
        tmp_pdf_path = tmp_pdf.name

        # Infer schema from PDF
        schema = detect_document_schema(tmp_pdf_path)
        response = {
            "schema": schema,
            "document_type_id": None  # Explicitly set to None when no document type is created
        }

        # Save as DocumentType if type and description provided
        if type and description:
            doc_type_create = DocumentTypeCreate(
                user_id=user_id,
                type=type,
                description=description,
                schema=schema
            )
            try:
                created_type = await service.create_document_type(doc_type_create)
                response["document_type_id"] = str(created_type.id)
            except Exception as e:
                logger.error(f"Error saving inferred schema: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save inferred schema: {str(e)}"
                )

        return SuccessResponse(data=response)
    except Exception as e:
        logger.error(f"Error inferring schema: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to infer schema: {str(e)}"
        )
    finally:
        os.remove(tmp_pdf.name)


@router.post(
    "/parse",
    response_model=SuccessResponse[dict],
    summary="Extract data from a PDF using a document type's schema",
    dependencies=[Depends(verify_api_key)]
)
async def parse_document_with_schema(
    document_type_id: UUID = Form(...),
    file: UploadFile = File(...),
    file_id: UUID = Form(None),
    service: DocumentTypeService = Depends(get_document_type_service),
    parsed_doc_service: ParsedDocumentService = Depends(get_parsed_document_service)
):
    """
    Extract data from a single-page PDF using the schema of the specified document type.
    Also stores the parsed data in the database, including file_id if provided.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    # Fetch the document type first to get user_id and validate existence
    document_type = await service.get_document_type(document_type_id)
    if not document_type or not document_type.schema:
        raise HTTPException(status_code=404, detail="Document type or schema not found.")

    # Check if this file was already parsed for this document type and user
    existing_parsed_doc = await parsed_doc_service.get_by_document_type_and_filename(
        document_type_id=document_type_id,
        user_id=document_type.user_id,
        original_file_name=file.filename
    )

    if existing_parsed_doc:
        return SuccessResponse(data={
            "parsed_document_id": str(existing_parsed_doc.id),
            "data": existing_parsed_doc.data,
            "message": f"Using existing parsed data for file '{file.filename}'"
        })

    tmp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        # Save uploaded PDF to temp file
        with tmp_pdf as f:
            shutil.copyfileobj(file.file, f)
        tmp_pdf_path = tmp_pdf.name

        # Parse document using the schema
        parsed_data = extract_document_data_with_schema(tmp_pdf_path, document_type.schema)
        
        # Create parsed document entry using the document_type's user_id
        parsed_doc = ParsedDocumentCreate(
            user_id=document_type.user_id,
            document_type_id=document_type_id,
            data=parsed_data,
            original_file_name=file.filename,
            file_id=file_id
        )
        
        try:
            saved_doc = await parsed_doc_service.create_parsed_document(parsed_doc)
            # Add the parsed document ID to the response
            parsed_data["parsed_document_id"] = str(saved_doc.id)
        except Exception as e:
            logger.error(f"Error saving parsed document: {str(e)}")
            # Continue even if saving fails - at least return the parsed data
            parsed_data["warning"] = "Failed to save parsed data"
        
        return SuccessResponse(data=parsed_data)
    except Exception as e:
        if (len(parsed_data) == 1):
            return SuccessResponse(data=parsed_data[0])
        logger.error(f"Error extracting data: {str(e)}")
        if(len(parsed_data)==1):
            return SuccessResponse(parsed_data[0])
        raise HTTPException(status_code=500, detail=f"Failed to extract data: {str(e)}")
    finally:
        os.remove(tmp_pdf.name)


@router.get(
    "/{user_id}/{type}",
    response_model=SuccessResponse[dict],
    summary="Get inferred schema by user ID and type",
    dependencies=[Depends(verify_api_key)]
)
async def get_inferred_schema_by_type(
    user_id: UUID,
    type: str,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """Get inferred schema by user ID and document type."""
    document_type = await service.get_by_type_and_user(type, user_id)
    if not document_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document type not found"
        )
    return SuccessResponse(data={
        "schema": document_type.schema,
        "document_type_id": str(document_type.id)
    })


@router.get(
    "/parse/{document_type_id}/{file_name}/{user_id}",
    response_model=SuccessResponse[dict],
    summary="Check if document was already parsed",
    dependencies=[Depends(verify_api_key)]
)
async def check_parsed_document(
    document_type_id: UUID,
    file_name: str,
    user_id: UUID,
    parsed_doc_service: ParsedDocumentService = Depends(get_parsed_document_service)
):
    """Check if a document was already parsed for this document type, user and filename."""
    existing_parsed_doc = await parsed_doc_service.get_by_document_type_and_filename(
        document_type_id=document_type_id,
        user_id=user_id,
        original_file_name=file_name
    )

    if not existing_parsed_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No existing parsed data found"
        )

    return SuccessResponse(data={
        "parsed_document_id": str(existing_parsed_doc.id),
        "data": existing_parsed_doc.data
    })


class UpdateSchemaRequest(BaseModel):
    schema: dict = Field(..., description="The new JSON schema to set for this document type.")

@router.patch(
    "/{document_type_id}/schema",
    response_model=SuccessResponse[DocumentType],
    summary="Update only the schema field of a document type",
    dependencies=[Depends(verify_api_key)]
)
async def update_document_type_schema(
    document_type_id: UUID,
    req: UpdateSchemaRequest,
    service: DocumentTypeService = Depends(get_document_type_service)
):
    """
    Update only the schema field of a document type.
    """
    # Use the existing update logic, but only pass the schema
    update_obj = DocumentTypeUpdate(schema=req.schema)
    updated = await service.update_document_type(document_type_id, update_obj)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document type with ID {document_type_id} not found"
        )
    return SuccessResponse(data=updated)


@router.delete(
    "/{document_type_id}/parsed-documents",
    response_model=SuccessResponse[bool],
    summary="Delete all parsed documents for a document type",
    dependencies=[Depends(verify_api_key)]
)
async def delete_parsed_documents_by_document_type(
    document_type_id: UUID,
    parsed_doc_service: ParsedDocumentService = Depends(get_parsed_document_service)
):
    """
    Delete all parsed documents for a specific document type.
    """
    try:
        result = await parsed_doc_service.delete_by_document_type(document_type_id)
        return SuccessResponse(data=result)
    except Exception as e:
        logger.error(f"Error deleting parsed documents by document type: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete parsed documents: {str(e)}"
        )