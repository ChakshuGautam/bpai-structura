# DocumentType API Implementation TODO

## Context and Overview

This TODO list is for implementing a DocumentType API to define document schemas and processing rules. A DocumentType allows defining post-processing pipelines during document conversion.

Key concepts:

- DocumentType includes type (string), description (string), schema (JSONSchema), and id (UUID)
- Supabase will be used as the backend database
- The API will integrate with the existing conversion pipeline
- We're using a modular architecture with separate routes, models, and services

## Folder Structure

core/
├── api/
│ ├── init.py
│ ├── app.py # Main FastAPI app factory
│ ├── routes/
│ │ ├── init.py
│ │ ├── convert.py # Existing convert endpoints
│ │ └── document_types.py # New document type endpoints
├── models/
│ ├── init.py
│ ├── document_type.py # Pydantic models for DocumentType
│ └── responses.py # Common response models
├── db/
│ ├── init.py
│ ├── client.py # Supabase client setup
│ └── repositories/
│ ├── init.py
│ └── document_type.py # DocumentType CRUD operations
└── services/
├── init.py
├── convert_service.py # Business logic for conversions
└── document_type_service.py # Business logic for document types

## Server Integration

- [x] Modify server.py to import document_types router from core/api/routes/document_types.py
- [x] Keep existing FastAPI app but include new routers
- [x] Extract common authentication logic to be reused across old and new endpoints

## Models Implementation

- [ ] Complete core/models/document_type.py with Pydantic models
- [ ] Define DocumentTypeBase, DocumentTypeCreate, DocumentTypeUpdate, and DocumentType models
- [ ] Implement appropriate validation for schema field

## Database Integration

- [x] Configure Supabase client in core/db/client.py (using supabase-py client instead of direct DB connection)
- [x] Implement CRUD operations in core/db/repositories/document_type.py
- [x] Create SQL migration script for document_types table (manually implemented via Supabase dashboard)

## API Routes

- [x] Implement document_types CRUD endpoints in core/api/routes/document_types.py
- [x] Ensure proper authentication is applied
- [x] Add validation and error handling

## Services Implementation

- [x] Create DocumentTypeService in core/services/document_type_service.py
- [x] Implement business logic for managing document types

## Document Processing

- [ ] Create DocumentTypeProcessor to process documents based on their type
- [ ] Integrate domain schema detection from get_domain_schema.py
- [ ] Modify existing conversion pipeline to use DocumentTypeProcessor when document_type_id is provided

## Testing

- [ ] Add unit tests for new models and repositories
- [ ] Add integration tests for the API endpoints
- [ ] Test end-to-end flow with different document types

## Documentation

- [ ] Update API documentation with new endpoints
- [ ] Add examples of using document types in conversion requests
- [ ] Document the schema format for defining document types

## Environment Configuration

- [x] Add SUPABASE_URL and SUPABASE_KEY to environment variables
- [ ] Update deployment configurations to include new environment variables

## Implementation Notes

### Server Integration Approach

We will take the following approach for server integration:

- Keep the existing FastAPI app instance in server.py
- Import the new document_types router from core/api/routes/document_types.py
- Use app.include_router() to add the new router to the existing app
- Maintain backward compatibility with all existing endpoints
- This approach allows incremental migration without disrupting existing functionality

### Authentication

- Extract the API key verification logic into a shared utility function
- Reuse this function across both old and new endpoints
- Ensure consistent authentication behavior throughout the application

### Document Processing

- The DocumentTypeProcessor will leverage the domain schema detection from get_domain_schema.py
- The conversion endpoints will be updated to accept an optional document_type_id parameter

# DocumentType Definition

A DocumentType is an abstraction that defines document-specific processing pipelines and schemas.

## Purpose

- Defines how documents should be processed during conversion

## Structure

- `id`: UUID (unique identifier)
- `type`: String (document type name, e.g., "Invoice", "Contract")
- `description`: String (human-readable description)
- `schema`: JSONSchema (defines the expected fields and their types)

## Functionality

- Acts as a template for document processing
- Defines validation rules for document data
- Specifies post-processing steps for extracted information
- Can be referenced during conversion API calls via `document_type_id`

## Usage Example

When converting a document:

1. Client can optionally provide a `document_type_id`
2. System uses the specified document type's schema to validate and structure the extracted data
3. Additional post-processing steps defined by the document type are applied
4. Results are formatted according to the document type's schema

> **Note**: This definition should be added near the beginning of the TODO.md file, directly after the "Context and Overview" section to provide essential context for the implementation tasks.
