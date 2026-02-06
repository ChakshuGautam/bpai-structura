from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, status, Form
from typing import List, Optional, Dict, Any
import asyncio
import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import logging
import time
from supabase import Client
import uuid
import aiohttp

from core.db.database import get_db

from ...models.ocr_models import (
    Image, ImageCreate, ImageUpdate, ImageWithEvaluations,
    Evaluation, EvaluationCreate, EvaluationUpdate, EvaluationWithDetails,
    PromptTemplate, PromptTemplateCreate, PromptTemplateUpdate,
    CSVImportRequest, CSVImportResponse,
    EvaluationStats, AccuracyDistribution,
    BatchProcessRequest, BatchProcessResponse,
    ImageFilter, PaginationParams, PaginatedResponse,
    PaginatedImagesResponse, PaginatedEvaluationsResponse,
    EvaluationProgress, EvaluationHistory, PromptVersionStats,
    Dataset, DatasetCreate, DatasetUpdate, DatasetWithImages, DatasetWithFiles,
    PromptFamily, PromptFamilyCreate, PromptFamilyWithVersions,
    PromptVersion, PromptVersionCreate, PromptVersionUpdate,
    EvaluationRun as EvaluationRunSchema, EvaluationRunCreate, EvaluationRunWithDetails,
    ComparisonResults, LiveProgressUpdate, PerformanceTrend,
    APIKey, APIKeyCreate, APIUsageStats, APILog,
    ProcessingStatus, DatasetStatus, PromptStatus, WordEvaluationCreate, APILogCreate,
    PDFExtractRequest, PDFExtractResponse, GenericEvalRequest, GenericEvalResponse,
    DeployInferResponse, DatasetFile
)
from . import crud
from ...models.orchestrator import OcrOrchestrator

router = APIRouter()

# Initialize OCR orchestrator lazily
ocr_orchestrator = None

def get_ocr_orchestrator():
    """Get OCR orchestrator instance, creating it if needed"""
    global ocr_orchestrator
    if ocr_orchestrator is None:
        try:
            logging.info("Initializing OCR orchestrator...")
            ocr_orchestrator = OcrOrchestrator()
            logging.info("OCR orchestrator initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OCR orchestrator: {str(e)}")
            raise
    return ocr_orchestrator

# Startup and shutdown events
@router.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    # Get the database client directly from the client module
    from core.db.client import supabase
    print("Database initialized")

# Health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "OCR Evaluation API is running"}

# Bulk import endpoint for pre-evaluated data
@router.post("/api/datasets/{dataset_id}/bulk-import-evaluations")
async def bulk_import_evaluations(
    dataset_id: int,
    data: Dict[str, Any],
    db: Client = Depends(get_db)
):
    """
    Bulk import dataset files with pre-computed evaluations.
    Expects JSON format with test_cases array containing:
    - image_url, reference_text, ocr_output
    - accuracy, correct_words, total_words
    - word_evaluations array
    """
    try:
        # Check dataset exists
        dataset = await crud.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        test_cases = data.get('test_cases', [])
        prompt_version_id = data.get('prompt_version_id')
        user_id = data.get('user_id', 'local-dev-user')

        if not test_cases:
            raise HTTPException(status_code=400, detail="No test_cases provided")
        if not prompt_version_id:
            raise HTTPException(status_code=400, detail="prompt_version_id required")

        imported_files = 0
        imported_evals = 0
        imported_words = 0

        for test_case in test_cases:
            # 1. Create dataset file (without dataset_id - use junction table)
            file_data = {
                "number": str(test_case.get('test_id', '')),
                "url": test_case['image_url'],
                "file_type": "image",
                "expected_output": test_case['reference_text'],
                "user_id": user_id,
                "metadata": {
                    "test_id": test_case.get('test_id'),
                    "original_accuracy": test_case.get('accuracy'),
                    "word_count": test_case.get('total_words')
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            file_result = db.table('dataset_files').insert(file_data).execute()
            file_id = file_result.data[0]['id'] if file_result.data else None

            if file_id:
                # Create association in junction table
                assoc_data = {
                    "dataset_id": dataset_id,
                    "file_id": file_id,
                    "created_at": datetime.utcnow().isoformat()
                }
                db.table('dataset_file_associations').insert(assoc_data).execute()
                imported_files += 1

            # 2. Create evaluation
            if file_id:
                eval_data = {
                    "file_id": file_id,
                    "prompt_version_id": prompt_version_id,
                    "model_id": test_case.get('model_id', 'machine-ocr'),
                    "processing_status": "completed",
                    "ocr_output": test_case['ocr_output'],
                    "accuracy": test_case['accuracy'],
                    "correct_words": test_case['correct_words'],
                    "total_words": test_case['total_words'],
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                eval_result = db.table('evaluations').insert(eval_data).execute()
                eval_id = eval_result.data[0]['id'] if eval_result.data else None
                if eval_id:
                    imported_evals += 1

                # 3. Create word evaluations
                if eval_id and test_case.get('word_evaluations'):
                    word_records = []
                    for word_eval in test_case['word_evaluations']:
                        word_records.append({
                            "evaluation_id": eval_id,
                            "reference_word": word_eval['reference_word'],
                            "transcribed_word": word_eval['transcribed_word'],
                            "match": word_eval['match'],
                            "reason_diff": word_eval.get('reason_diff', ''),
                            "word_position": word_eval.get('word_position', 0)
                        })

                    if word_records:
                        db.table('word_evaluations').insert(word_records).execute()
                        imported_words += len(word_records)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "imported_files": imported_files,
            "imported_evaluations": imported_evals,
            "imported_word_evaluations": imported_words
        }

    except Exception as e:
        logging.error(f"Bulk import error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# OpenAPI spec endpoints
@router.get("/api/openapi.yaml")
async def get_openapi_yaml():
    """Get OpenAPI specification in YAML format"""
    import yaml
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=router.title,
        version=router.version,
        description=router.description,
        routes=router.routes,
    )
    
    from fastapi.responses import Response
    yaml_content = yaml.dump(openapi_schema, default_flow_style=False)
    return Response(content=yaml_content, media_type="application/x-yaml")

# Image endpoints
@router.get("/api/images", response_model=PaginatedImagesResponse)
async def get_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    has_evaluations: Optional[bool] = Query(None),
    processing_status: Optional[str] = Query(None),
    prompt_version: Optional[str] = Query(None),
    accuracy_min: Optional[float] = Query(None, ge=0, le=100),
    accuracy_max: Optional[float] = Query(None, ge=0, le=100),
    db: Client = Depends(get_db)
):
    """Get paginated list of images with optional filters"""
    filters = ImageFilter(
        has_evaluations=has_evaluations,
        processing_status=processing_status,
        prompt_version=prompt_version,
        accuracy_min=accuracy_min,
        accuracy_max=accuracy_max
    )
    pagination = PaginationParams(skip=skip, limit=limit)
    
    images, total = await crud.get_images(db, filters, pagination)
    
    return PaginatedImagesResponse(
        items=images,
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + limit < total
    )

@router.get("/api/images/{image_id}", response_model=ImageWithEvaluations)
async def get_image(image_id: int, db: Client = Depends(get_db)):
    """Get a specific image with its evaluations"""
    image = await crud.get_image(db, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image

@router.post("/api/images", response_model=Image)
async def create_image(image: ImageCreate, db: Client = Depends(get_db)):
    """Create a new image"""
    # Check if image with this number already exists
    existing = await crud.get_image_by_number(db, image.number)
    if existing:
        raise HTTPException(status_code=400, detail="Image with this number already exists")
    
    return await crud.create_image(db, image)

@router.put("/api/images/{image_id}", response_model=Image)
async def update_image(
    image_id: int, 
    image_update: ImageUpdate, 
    db: Client = Depends(get_db)
):
    """Update an existing image"""
    image = await crud.update_image(db, image_id, image_update)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image

@router.delete("/api/images/{image_id}")
async def delete_image(image_id: int, db: Client = Depends(get_db)):
    """Delete an image and all its evaluations"""
    success = await crud.delete_image(db, image_id)
    if not success:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"message": "Image deleted successfully"}

@router.post("/api/images/import-csv", response_model=CSVImportResponse)
async def import_images_csv(
    file: UploadFile = File(...),
    overwrite_existing: bool = Query(False),
    db: Client = Depends(get_db)
):
    """Import images from CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Create a temporary file with proper extension
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        # Save uploaded file temporarily
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Import data
        result = await crud.import_csv_data(db, temp_path, overwrite_existing)
        return CSVImportResponse(**result)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/api/images/{dataset_id}/import-csv", response_model=CSVImportResponse)
async def import_images_csv_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    overwrite_existing: bool = Query(False),
    db: Client = Depends(get_db)
):
    """Import images from CSV file and associate them with a dataset (legacy)"""
    # logging.info(f"Hit /api/images/{dataset_id}/import-csv with dataset_id={dataset_id}")
    # First check if dataset exists
    dataset = await crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Create a temporary file with proper extension
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        # Save uploaded file temporarily
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Import data
        result = await crud.import_csv_data_into_dataset(db, temp_path, dataset_id, overwrite_existing)
        return CSVImportResponse(**result)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/api/datasets/{dataset_id}/import-csv", response_model=CSVImportResponse)
async def import_dataset_files_csv(
    dataset_id: int,
    file: UploadFile = File(...),
    overwrite_existing: bool = Query(False),
    db: Client = Depends(get_db)
):
    """Import dataset files from CSV file and associate them with a dataset (new format)"""
    # First check if dataset exists
    dataset = await crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Verify that the dataset belongs to the current user
    # Note: In a real implementation, you'd get the current user from auth context
    # For now, we'll rely on the dataset's user_id which should be set correctly
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Create a temporary file with proper extension
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        # Save uploaded file temporarily
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Import data using new dataset files function
        result = await crud.import_csv_data_into_dataset_files(db, temp_path, dataset_id, overwrite_existing)
        return CSVImportResponse(**result)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/api/images/export-csv")
async def export_images_csv(db: Client = Depends(get_db)):
    """Export images to CSV format"""
    import pandas as pd
    from io import StringIO
    from fastapi.responses import StreamingResponse
    
    # Get all images with their latest evaluations
    images = await crud.get_all_images_with_evaluations(db)
    
    # Convert to CSV format
    csv_data = []
    for image in images:
        latest_eval = image.evaluations[0] if image.evaluations else None

        # Escape newlines in text fields to prevent multi-row CSV issues
        expected_text = image.expected_text.replace('\n', '\\n').replace('\r', '\\r') if image.expected_text else ''
        ocr_output = latest_eval.ocr_output.replace('\n', '\\n').replace('\r', '\\r') if latest_eval and latest_eval.ocr_output else ''

        row = {
            '#': image.id,
            'Link': image.file_path,
            'Text': expected_text,
            'Correctness': '',  # This would need to be populated based on your needs
            'OCR Output (Gemini - Flash)': ocr_output,
            'OCR Output with Text Priming (Gemini - Flash)': ''  # Additional field if needed
        }
        csv_data.append(row)
    
    # Create DataFrame and convert to CSV
    df = pd.DataFrame(csv_data)
    
    # Create StringIO buffer
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=images_export.csv"}
    )

# Evaluation endpoints
@router.get("/api/evaluations", response_model=PaginatedEvaluationsResponse)
async def get_evaluations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    image_id: Optional[int] = Query(None),
    prompt_version_id: Optional[int] = Query(None),
    db: Client = Depends(get_db)
):
    """Get paginated list of evaluations"""
    pagination = PaginationParams(skip=skip, limit=limit)
    evaluations, total = await crud.get_evaluations(db, image_id, prompt_version_id, pagination)
    
    return PaginatedEvaluationsResponse(
        items=evaluations,
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + limit < total
    )

# Specific routes must come before parameterized routes
@router.get("/api/evaluations/active", response_model=List[EvaluationProgress])
async def get_active_evaluations(db: Client = Depends(get_db)):
    """Get all currently processing evaluations"""
    result = db.table('evaluations').select('*').or_('processing_status.eq.pending,processing_status.eq.processing').execute()
    
    active_evaluations = result.data
    
    return [
        EvaluationProgress(
            evaluation_id=eval['id'],
            processing_status=eval['processing_status'],
            progress_percentage=eval['progress_percentage'] or 0,
            current_step=eval['current_step'],
            estimated_completion=eval['estimated_completion'],
            created_at=eval['created_at'],
            updated_at=eval['updated_at']
        )
        for eval in active_evaluations
    ]

@router.get("/api/evaluations/history", response_model=List[EvaluationHistory])
async def get_evaluation_history(
    prompt_version: Optional[str] = Query(None),
    db: Client = Depends(get_db)
):
    """Get evaluation history grouped by prompt version"""
    # Get all prompt versions with their evaluations
    if prompt_version:
        # Filter by specific prompt version
        versions_result = db.table('evaluations').select('prompt_version').eq('prompt_version', prompt_version).execute()
    else:
        # Get all versions
        versions_result = db.table('evaluations').select('prompt_version').execute()
    
    # Get unique versions
    versions = list(set(v['prompt_version'] for v in versions_result.data if v['prompt_version']))
    
    history = []
    for version in versions:
        # Get evaluations for this version
        evaluations_result = db.table('evaluations').select('*').eq('prompt_version', version).execute()
        evaluations = evaluations_result.data
        
        # Calculate average accuracy
        successful_evals = [e for e in evaluations if e['processing_status'] == "success" and e['accuracy']]
        avg_accuracy = None
        if successful_evals:
            avg_accuracy = sum(e['accuracy'] for e in successful_evals) / len(successful_evals)
        
        # Get prompt template if it exists
        template_result = db.table('prompt_templates').select('*').eq('version', version).execute()
        prompt_template = template_result.data[0] if template_result.data else None
        
        history.append(EvaluationHistory(
            prompt_version=version,
            evaluations=evaluations[:50],  # Limit to recent 50
            total_count=len(evaluations),
            avg_accuracy=avg_accuracy,
            prompt_template=prompt_template
        ))
    
    return history

@router.get("/api/evaluations/{evaluation_id}", response_model=EvaluationWithDetails)
async def get_evaluation(evaluation_id: int, db: Client = Depends(get_db)):
    """Get a specific evaluation with full details"""
    evaluation = await crud.get_evaluation(db, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation

@router.post("/api/evaluations", response_model=Evaluation)
async def create_evaluation(
    evaluation: EvaluationCreate, 
    background_tasks: BackgroundTasks,
    db: Client = Depends(get_db)
):
    """Create a new evaluation and queue it for processing"""
    # Determine which ID to use and verify the target exists
    if evaluation.file_id is not None:
        # Use file_id (new approach)
        file = await crud.get_dataset_file(db, evaluation.file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
    elif evaluation.image_id is not None:
        # Use image_id (backward compatibility)
        image = await crud.get_image(db, evaluation.image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
    else:
        raise HTTPException(status_code=400, detail="Either file_id or image_id must be provided")
    
    # Create evaluation record
    db_evaluation = await crud.create_evaluation(db, evaluation)
    
    # Queue background processing
    background_tasks.add_task(process_evaluation_background, db_evaluation['id'])
    
    return db_evaluation

@router.post("/api/evaluations/batch", response_model=BatchProcessResponse)
async def batch_process_evaluations(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    db: Client = Depends(get_db)
):
    """Create and process multiple evaluations"""
    queued_count = 0
    
    try:
        # Verify each image exists and create evaluations
        for image_id in request.image_ids:
            # Check if evaluation already exists
            existing_eval = db.table('evaluations').select('*').eq('image_id', image_id).eq('prompt_version', request.prompt_version).execute()
            if existing_eval.data and not request.force_reprocess:
                logging.info(f"Skipping image {image_id} - evaluation already exists")
                continue

            try:
                # Create evaluation record directly in Supabase with only the required fields
                eval_data = {
                    'image_id': image_id,
                    'prompt_version': request.prompt_version,
                    'processing_status': 'pending',
                    'progress_percentage': 0,
                    'current_step': 'Queued for processing',
                    'accuracy': 0,  # Default value
                    'correct_words': 0,  # Default value
                    'total_words': 0,  # Default value
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                logging.info(f"Creating evaluation with data: {eval_data}")
                result = db.table('evaluations').insert(eval_data).execute()
                
                if result.data:
                    evaluation_id = result.data[0]['id']
                    # Queue background processing
                    background_tasks.add_task(process_evaluation_background, evaluation_id)
                    queued_count += 1
                    logging.info(f"Created evaluation {evaluation_id} for image {image_id}")
                else:
                    logging.error(f"Failed to create evaluation for image {image_id}")

            except Exception as e:
                error_msg = str(e)
                if isinstance(e, dict):
                    error_msg = json.dumps(e)
                logging.error(f"Error creating evaluation for image {image_id}: {error_msg}")
                continue
        
        response = BatchProcessResponse(
            queued_count=queued_count,
            message=f"Queued {queued_count} evaluations for processing"
        )
        logging.info(f"Batch processing complete: {response}")
        return response
        
    except Exception as e:
        error_msg = str(e)
        if isinstance(e, dict):
            error_msg = json.dumps(e)
        logging.error(f"Error in batch processing: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

# Progress and Status endpoints

@router.get("/api/evaluations/{evaluation_id}/progress", response_model=EvaluationProgress)
async def get_evaluation_progress(evaluation_id: int, db: Client = Depends(get_db)):
    """Get real-time progress of an evaluation"""
    evaluation = await crud.get_evaluation(db, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return EvaluationProgress(
        evaluation_id=evaluation.id,
        processing_status=evaluation.processing_status,
        progress_percentage=evaluation.progress_percentage or 0,
        current_step=evaluation.current_step,
        estimated_completion=evaluation.estimated_completion,
        created_at=evaluation.created_at,
        updated_at=evaluation.updated_at
    )

# History and Prompt Version endpoints

@router.get("/api/prompt-versions/stats", response_model=List[PromptVersionStats])
async def get_prompt_version_stats(db: Client = Depends(get_db)):
    """Get statistics for each prompt version"""
    # Get all evaluations with prompt version details
    result = db.table('evaluations').select('*, prompt_version:prompt_versions(*)').execute()
    evaluations = result.data
    
    # Group by prompt version
    stats_by_version = {}
    for eval in evaluations:
        # Get the prompt version string from the joined prompt_versions table
        prompt_version_data = eval.get('prompt_version', [])
        if isinstance(prompt_version_data, list) and prompt_version_data:
            version = prompt_version_data[0].get('version')
        elif isinstance(prompt_version_data, dict):
            version = prompt_version_data.get('version')
        else:
            # Fallback: try to get version from prompt_version_id
            prompt_version_id = eval.get('prompt_version_id')
            if prompt_version_id:
                version_result = db.table('prompt_versions').select('version').eq('id', prompt_version_id).single().execute()
                version = version_result.data.get('version') if version_result.data else None
            else:
                version = None
        
        if not version:
            continue  # Skip evaluations without a valid prompt version
            
        if version not in stats_by_version:
            stats_by_version[version] = {
                'total': 0,
                'successful': 0,
                'accuracies': [],
                'created_at': eval['created_at'],
                'latest_evaluation': eval['created_at']
            }
        
        stats = stats_by_version[version]
        stats['total'] += 1
        
        if eval['processing_status'] == 'success':
            stats['successful'] += 1
            if eval['accuracy'] is not None:
                stats['accuracies'].append(eval['accuracy'])
        
        # Update timestamps
        if eval['created_at'] < stats['created_at']:
            stats['created_at'] = eval['created_at']
        if eval['created_at'] > stats['latest_evaluation']:
            stats['latest_evaluation'] = eval['created_at']
    
    # Convert to list of PromptVersionStats
    return [
        PromptVersionStats(
            version=version,
            total_evaluations=stats['total'],
            successful_evaluations=stats['successful'],
            avg_accuracy=sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else None,
            created_at=stats['created_at'],
            latest_evaluation=stats['latest_evaluation']
        )
        for version, stats in stats_by_version.items()
    ]

# Prompt Template endpoints
@router.get("/api/prompt-templates", response_model=List[PromptTemplate])
async def get_prompt_templates(db: Client = Depends(get_db)):
    """Get all prompt templates"""
    return await crud.get_prompt_templates(db)

@router.get("/api/prompt-templates/active", response_model=PromptTemplate)
async def get_active_prompt_template(db: Client = Depends(get_db)):
    """Get the currently active prompt template"""
    template = await crud.get_active_prompt_template(db)
    if not template:
        raise HTTPException(status_code=404, detail="No active prompt template found")
    return template

@router.post("/api/prompt-templates", response_model=PromptTemplate)
async def create_prompt_template(
    template: PromptTemplateCreate, 
    db: Client = Depends(get_db)
):
    """Create a new prompt template"""
    return await crud.create_prompt_template(db, template)

@router.put("/api/prompt-templates/{template_id}", response_model=PromptTemplate)
async def update_prompt_template(
    template_id: int,
    template_update: PromptTemplateUpdate,
    db: Client = Depends(get_db)
):
    """Update a prompt template"""
    # Implementation would be similar to image update
    # For now, just return error
    raise HTTPException(status_code=501, detail="Not implemented yet")

# CSV Import endpoints
@router.post("/api/import/csv", response_model=CSVImportResponse)
async def import_csv_file(
    file: UploadFile = File(...),
    overwrite_existing: bool = Query(False),
    db: Client = Depends(get_db)
):
    """Import images from CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Create a temporary file with proper extension
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        # Save uploaded file temporarily
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Import data
        result = await crud.import_csv_data(db, temp_path, overwrite_existing)
        return CSVImportResponse(**result)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/api/import/csv/file-path", response_model=CSVImportResponse)
async def import_csv_from_path(
    request: CSVImportRequest,
    db: Client = Depends(get_db)
):
    """Import images from CSV file path"""
    if not request.file_path:
        # Default to the existing CSV file
        request.file_path = "images.csv"
    
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    result = await crud.import_csv_data(db, request.file_path, request.overwrite_existing)
    return CSVImportResponse(**result)

# Statistics endpoints
@router.get("/api/stats/evaluations", response_model=EvaluationStats)
async def get_evaluation_statistics(db: Client = Depends(get_db)):
    """Get evaluation statistics"""
    stats = await crud.get_evaluation_stats(db)
    return EvaluationStats(**stats)

@router.get("/api/stats/accuracy-distribution", response_model=AccuracyDistribution)
async def get_accuracy_distribution(db: Client = Depends(get_db)):
    """Get accuracy distribution"""
    distribution = await crud.get_accuracy_distribution(db)
    return AccuracyDistribution(**distribution)

# File serving for images
@router.get("/api/images/{image_id}/file")
async def get_image_file(image_id: int, db: Client = Depends(get_db)):
    """Serve the actual image file"""
    image = await crud.get_image(db, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not image.local_path or not os.path.exists(image.local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(image.local_path)

# PDF Extraction endpoint
@router.post("/api/pdf-extract", response_model=PDFExtractResponse)
async def extract_from_pdf(
    pdf_url: str = Form(..., description="URL to the PDF file to extract from"),
    prompt: str = Form(..., description="Prompt to use for extraction")
):
    """Extract structured information from a PDF using Gemini OCR"""
    try:
        # Get OCR orchestrator
        orchestrator = get_ocr_orchestrator()
        images_dir = orchestrator.images_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Starting PDF extraction from URL: {pdf_url}")
        
        # Convert Google Drive URLs if needed (reuse logic from orchestrator)
        original_url = pdf_url
        pdf_url = orchestrator._convert_google_drive_url(pdf_url)
        if pdf_url != original_url:
            logging.info(f"Converted Google Drive URL: {original_url} -> {pdf_url}")
        
        # Validate URL
        if not orchestrator._validate_url(pdf_url):
            raise HTTPException(status_code=400, detail="Invalid PDF URL")
        
        # Generate unique filename in images_dir
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pdf_api_{timestamp}.pdf"
        file_path = images_dir / filename
        
        # Download PDF to images_dir
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if 'text/html' in content_type.lower():
                        raise HTTPException(status_code=400, detail="URL returns HTML content, not a PDF file")
                    
                    # Write PDF content to file_path
                    with open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
            
            # Validate PDF file
            if not orchestrator._validate_pdf_file(str(file_path)):
                raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")
            
            logging.info(f"PDF downloaded successfully to: {file_path}")
            
            # Process PDF using Gemini OCR
            result = orchestrator.ocr.extract_from_pdf_with_model(str(file_path), prompt, 'gemini-2.5-flash-preview-05-20')
            
            if 'error' in result:
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {result['error']}")
            
            # Return successful response
            return PDFExtractResponse(
                structured_output=result.get('structured_output', {}),
                processing_method=result.get('processing_method', 'direct_pdf_upload'),
                tokens_used=result.get('tokens_used'),
                success=True,
                error_message=None
            )
            
        finally:
            # Clean up the downloaded file
            if file_path.exists():
                os.remove(file_path)
                logging.info(f"Cleaned up file: {file_path}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error in PDF extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

# Generic Evaluation endpoint
@router.post("/api/eval/infer", response_model=GenericEvalResponse)
async def generic_evaluation(
    request: GenericEvalRequest,
    db: Client = Depends(get_db)
):
    try:
        eval_id = request.eval_config_id
        input_data = request.input

        logging.info(f"Starting generic evaluation for eval_id: {eval_id}")

        # Get evaluation record with prompt_version_id and file_id
        eval_result = db.table('evaluations').select('prompt_version_id, file_id').eq('id', eval_id).execute()
        if not eval_result.data:
            raise HTTPException(status_code=404, detail=f"Evaluation {eval_id} not found")

        evaluation = eval_result.data[0]
        prompt_version_id = evaluation.get('prompt_version_id')
        file_id = evaluation.get('file_id')

        if not prompt_version_id:
            raise HTTPException(status_code=400, detail=f"Evaluation {eval_id} has no prompt_version_id")

        if not file_id:
            raise HTTPException(status_code=400, detail=f"Evaluation {eval_id} has no file_id")

        # Get prompt_text from prompt_versions table
        prompt_result = db.table('prompt_versions').select('prompt_text').eq('id', prompt_version_id).execute()
        if not prompt_result.data:
            raise HTTPException(status_code=404, detail=f"Prompt version {prompt_version_id} not found")

        prompt_text = prompt_result.data[0].get('prompt_text')
        if not prompt_text:
            raise HTTPException(status_code=400, detail=f"Prompt version {prompt_version_id} has no prompt_text")

        # Get file_type from dataset_files table
        file_result = db.table('dataset_files').select('file_type, url, expected_output').eq('id', file_id).execute()
        if not file_result.data:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

        file_data = file_result.data[0]
        file_type = file_data.get('file_type', 'image')
        file_url = file_data.get('url')
        expected_output_db = file_data.get('expected_output')
        if not file_url:
            raise HTTPException(status_code=400, detail=f"File {file_id} has no URL")

        # Get dataset information to determine output_type/input_type
        dataset_output_type = "text"  # Default fallback
        dataset_input_type = None
        try:
            dataset_result = db.table('dataset_file_associations').select('dataset_id').eq('file_id', file_id).execute()
            if dataset_result.data:
                dataset_id = dataset_result.data[0].get('dataset_id')
                if dataset_id:
                    dataset_info_result = db.table('datasets').select('output_type', 'input_type').eq('id', dataset_id).execute()
                    if dataset_info_result.data:
                        dataset_output_type = dataset_info_result.data[0].get('output_type', 'text')
                        dataset_input_type = dataset_info_result.data[0].get('input_type', None)
                        logging.info(f"Found dataset output_type: {dataset_output_type}, input_type: {dataset_input_type}")
        except Exception as e:
            logging.warning(f"Could not get dataset output_type/input_type for file_id {file_id}: {str(e)}")

        orchestrator = get_ocr_orchestrator()

        # If input is provided, use its url and reference_text as needed
        if input_data and isinstance(input_data, dict) and "url" in input_data:
            file_url = input_data.get("url")
            # Use reference_text only if output_type or input_type is 'image'
            expected_output = None
            if (dataset_input_type and str(dataset_input_type).lower() == "image"):
                expected_output = input_data.get("reference_text")
            # Call orchestrator
            result = await orchestrator.process_single_evaluation(
                file_url,
                expected_output,
                "input",  # Use a dummy file_id or identifier
                prompt_text,
                None,  # file_id
                (dataset_input_type or "IMAGE").upper() if dataset_input_type else file_type.upper() if file_type else None
            )
            if (dataset_input_type and str(dataset_input_type).lower() == "image"):
                # Compose output as required for images
                if result.get('success'):
                    evaluation_data = result.get('evaluation', {})
                    output = {
                        "full_text": evaluation_data.get('ocr_output', ''),
                        "word_evaluations": evaluation_data.get('word_evaluations', []),
                        "total_words": evaluation_data.get('total_words', 0),
                        "correct_words": evaluation_data.get('correct_words', 0),
                        "accuracy": evaluation_data.get('accuracy', 0),
                    }
                    return GenericEvalResponse(
                        output=output,
                        evaluations=evaluation_data.get('word_evaluations', []),
                        overall_accuracy=evaluation_data.get('accuracy', 100.0),
                        tokens_used=result.get('tokens_used', 0),
                        success=True,
                        error_message=None
                    )
                else:
                    return GenericEvalResponse(
                        output={"full_text": "", "word_evaluations": [], "total_words": 0, "correct_words": 0, "accuracy": 0},
                        evaluations=[],
                        overall_accuracy=None,
                        tokens_used=result.get('tokens_used', 0),
                        success=False,
                        error_message=result.get('error', 'Unknown error')
                    )
            else:
                # Fallback to original output logic for non-image types
                if result.get('success'):
                    evaluation_data = result.get('evaluation', {})
                    raw_output = evaluation_data.get('ocr_output', '')
                    formatted_output = raw_output
                    if dataset_output_type == "json":
                        try:
                            if isinstance(raw_output, str):
                                import json
                                formatted_output = json.loads(raw_output)
                            else:
                                formatted_output = raw_output
                        except (json.JSONDecodeError, TypeError):
                            formatted_output = raw_output
                    else:
                        formatted_output = str(raw_output) if raw_output is not None else ""
                    # Format evaluations based on file_type
                    raw_evaluations = evaluation_data.get('word_evaluations', [])
                    formatted_evaluations = raw_evaluations
                    return GenericEvalResponse(
                        output=formatted_output,
                        evaluations=formatted_evaluations,
                        overall_accuracy=evaluation_data.get('accuracy', 100.0),
                        tokens_used=result.get('tokens_used', 0),
                        success=True,
                        error_message=None
                    )
                else:
                    return GenericEvalResponse(
                        output="" if dataset_output_type == "text" else {},
                        evaluations=[],
                        overall_accuracy=None,
                        tokens_used=result.get('tokens_used', 0),
                        success=False,
                        error_message=result.get('error', 'Unknown error')
                    )

        # Fallback to current DB-based logic if no input is provided
        # Use expected_output from DB
        result = await orchestrator.process_single_evaluation(
            file_url,
            expected_output_db,
            str(file_id),  # Use file_id as number
            prompt_text,
            file_id,
            file_type.upper() if file_type else None
        )
        if result.get('success'):
            evaluation_data = result.get('evaluation', {})
            raw_output = evaluation_data.get('ocr_output', '')
            formatted_output = raw_output
            if dataset_output_type == "json":
                try:
                    if isinstance(raw_output, str):
                        import json
                        formatted_output = json.loads(raw_output)
                    else:
                        formatted_output = raw_output
                except (json.JSONDecodeError, TypeError):
                    formatted_output = raw_output
            else:
                formatted_output = str(raw_output) if raw_output is not None else ""
            # Format evaluations based on file_type
            raw_evaluations = evaluation_data.get('word_evaluations', [])
            formatted_evaluations = raw_evaluations
            return GenericEvalResponse(
                output=formatted_output,
                evaluations=formatted_evaluations,
                overall_accuracy=evaluation_data.get('accuracy', 100.0),
                tokens_used=result.get('tokens_used', 0),
                success=True,
                error_message=None
            )
        else:
            return GenericEvalResponse(
                output="" if dataset_output_type == "text" else {},
                evaluations=[],
                overall_accuracy=None,
                tokens_used=result.get('tokens_used', 0),
                success=False,
                error_message=result.get('error', 'Unknown error')
            )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error in generic evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generic evaluation failed: {str(e)}")

# Background task functions
async def process_evaluation_background(evaluation_id: int):
    """Background task to process an evaluation"""
    from core.db.database import get_db
    
    # Get database connection
    db = None
    async for database in get_db():
        db = database
        break
    
    if not db:
        logging.error("Failed to get database connection")
        return
        
    try:
        # Get evaluation with file data first (new approach)
        result = db.table('evaluations').select('*,file:dataset_files(*)').eq('id', evaluation_id).execute()
        evaluation = result.data[0] if result.data else None
        
        if evaluation and evaluation.get('file'):
            # Use file data (new approach)
            file_data = evaluation.get('file', [])
            file = file_data[0] if isinstance(file_data, list) else file_data
            
            # Map file data to image format for compatibility
            image = {
                'url': file.get('url'),
                'human_evaluation_text': file.get('metadata', {}).get('human_evaluation_text', ''),
                'number': file.get('number'),
                'expected_output': file.get('expected_output')
            }
        else:
            # Fallback to image data (backward compatibility)
            result = db.table('evaluations').select('*,image:images(*)').eq('id', evaluation_id).execute()
            if not result.data or not result.data[0]:
                logging.error(f"Evaluation {evaluation_id} not found")
                return
                
            evaluation = result.data[0]
            image_data = evaluation.get('image', [])
            if not image_data:
                logging.error(f"No image or file data found for evaluation {evaluation_id}")
                raise Exception("No image or file data found")
                
            image = image_data[0] if isinstance(image_data, list) else image_data
        
        # Update status to processing with progress
        await crud.update_evaluation(
            db, 
            evaluation_id, 
            EvaluationUpdate(
                processing_status="processing",
                progress_percentage=10,
                current_step="Initializing OCR processing"
            )
        )
        
        # Update progress: downloading image
        await crud.update_evaluation(
            db, 
            evaluation_id, 
            EvaluationUpdate(
                progress_percentage=30,
                current_step="Downloading image"
            )
        )
        
        # Process the evaluation using OCR orchestrator
        orchestrator = get_ocr_orchestrator()

        # Fetch the prompt_text for this evaluation's prompt_version_id
        prompt_text = None
        prompt_version_id = evaluation.get('prompt_version_id')
        
        if prompt_version_id:
            # Fetch prompt version by ID directly
            prompt_version_result = db.table('prompt_versions').select('*').eq('id', prompt_version_id).single().execute()
            if prompt_version_result.data:
                prompt_version_obj = prompt_version_result.data
                prompt_text = prompt_version_obj.get('prompt_text')
                logging.info(f"[EVAL {evaluation_id}] Using prompt_version_id: {prompt_version_id}")
                if prompt_text:
                    logging.info(f"[EVAL {evaluation_id}] prompt_text (first 200 chars): {prompt_text[:200]}")
                else:
                    logging.warning(f"[EVAL {evaluation_id}] No prompt_text found for prompt_version_id: {prompt_version_id}")
            else:
                logging.error(f"[EVAL {evaluation_id}] Prompt version not found for ID: {prompt_version_id}")
        else:
            logging.error(f"[EVAL {evaluation_id}] No prompt_version_id found in evaluation record")
        
        # Update progress: running OCR
        await crud.update_evaluation(
            db, 
            evaluation_id, 
            EvaluationUpdate(
                progress_percentage=60,
                current_step="Running OCR analysis"
            )
        )
        
        # Get dataset information to determine input_type
        dataset_input_type = None
        if evaluation.get('file_id'):
            # Get the dataset through the file association
            file_result = db.table('dataset_files').select('*, dataset_file_associations(dataset_id)').eq('id', evaluation.get('file_id')).execute()
            if file_result.data and file_result.data[0]:
                file_data = file_result.data[0]
                associations = file_data.get('dataset_file_associations', [])
                if associations:
                    dataset_id = associations[0].get('dataset_id')
                    if dataset_id:
                        dataset_result = db.table('datasets').select('input_type').eq('id', dataset_id).execute()
                        if dataset_result.data and dataset_result.data[0]:
                            dataset_input_type = dataset_result.data[0].get('input_type')
                            logging.info(f"[EVAL {evaluation_id}] Dataset input_type: {dataset_input_type}")
        
        # Extract model_id and model_info from evaluation record
        model_id = evaluation.get('model_id')
        model_info = evaluation.get('model_info')
        
        start_time = time.time()
        result = await orchestrator.process_single_evaluation(
            image.get('url'),
            image.get('expected_output', image.get('human_evaluation_text', '')),
            image.get('number'),
            prompt_text,
            evaluation.get('file_id'),
            dataset_input_type,
            model_id,  # Pass model_id
            model_info  # Pass model_info
        )
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Update progress: analyzing results
        await crud.update_evaluation(
            db, 
            evaluation_id, 
            EvaluationUpdate(
                progress_percentage=90,
                current_step="Analyzing results"
            )
        )
        
        if result.get('success'):
            # Update evaluation with results
            word_evaluations = []
            evaluation_data = result.get('evaluation', {})
            
            for word_eval in evaluation_data.get('word_evaluations', []):
                word_evaluations.append(WordEvaluationCreate(
                    reference_word=word_eval.get('reference_word', ''),
                    transcribed_word=word_eval.get('transcribed_word'),
                    match=word_eval.get('match', False),
                    reason_diff=word_eval.get('reason_diff', ''),
                    word_position=word_eval.get('word_position', 0)
                ))
            
            update_data = EvaluationUpdate(
                ocr_output=evaluation_data.get('ocr_output', ''),
                accuracy=evaluation_data.get('accuracy', 0),
                correct_words=evaluation_data.get('correct_words', 0),
                total_words=evaluation_data.get('total_words', 0),
                processing_status="success",
                progress_percentage=100,
                current_step="Completed",
                word_evaluations=word_evaluations
            )
            
            await crud.update_evaluation(db, evaluation_id, update_data)

            # Add API log entry
            try:
                # Get the prompt version string for logging
                prompt_version_str = None
                if prompt_version_id:
                    prompt_version_result = db.table('prompt_versions').select('version').eq('id', prompt_version_id).single().execute()
                    if prompt_version_result.data:
                        prompt_version_str = prompt_version_result.data.get('version')
                
                log_entry = APILogCreate(
                    image_url=image.get('url'),
                    ocr_output=evaluation_data.get('ocr_output', ''),
                    prompt_version=prompt_version_str,
                    log_metadata={
                        "tokens_used": result.get('tokens_used', 0),
                        "latency_ms": latency_ms
                    }
                )
                await crud.create_api_log(db, log_entry)
            except Exception as e:
                logging.error(f"Failed to create API log for evaluation {evaluation_id}: {str(e)}")
        else:
            # Update with error
            await crud.update_evaluation(
                db,
                evaluation_id,
                EvaluationUpdate(
                    processing_status="failed",
                    progress_percentage=0,
                    current_step="Failed",
                    error_message=result.get('error', 'Unknown error')
                )
            )
    
    except Exception as e:
        logging.error(f"Error processing evaluation {evaluation_id}: {str(e)}")
        # Update with error
        await crud.update_evaluation(
            db,
            evaluation_id,
            EvaluationUpdate(
                processing_status="failed",
                progress_percentage=0,
                current_step="Failed",
                error_message=str(e)
            )
        )

async def process_evaluation_run_background(run_id: int):
    """Background task to process an evaluation run (A/B test)"""
    from core.db.database import get_db
    
    # Get database connection
    async for db in get_db():
        try:
            # Get the evaluation run
            evaluation_run = await crud.get_evaluation_run(db, run_id)
            if not evaluation_run:
                return
            
            # Update run status to processing
            db.table('evaluation_runs').update({
                'status': 'processing',
                'progress_percentage': 0,
                'current_step': 'Initializing evaluation run'
            }).eq('id', run_id).execute()
            
            # Get all files from all datasets in this run (prefer files, fallback to images for legacy)
            all_files = []
            for dataset in evaluation_run.get('datasets', []):
                if 'files' in dataset and dataset['files']:
                    all_files.extend(dataset['files'])
                elif 'images' in dataset and dataset['images']:
                    all_files.extend(dataset['images'])
            
            total_files = len(all_files)
            if total_files == 0:
                # No files to process
                db.table('evaluation_runs').update({
                    'status': 'failed',
                    'current_step': 'No files found in datasets'
                }).eq('id', run_id).execute()
                return
            
            # Get prompt configurations for this run
            prompt_configs = evaluation_run.get('prompt_configurations', [])
            if not prompt_configs:
                db.table('evaluation_runs').update({
                    'status': 'failed',
                    'current_step': 'No prompt configurations found'
                }).eq('id', run_id).execute()
                return
            
            # Process each file with each prompt configuration
            processed_count = 0
            orchestrator = get_ocr_orchestrator()
            
            for file in all_files:
                for prompt_config in prompt_configs:
                    try:
                        # Determine if this file has a file_id (new approach) or image_id (legacy)
                        file_id = file.get('id') if 'expected_output' in file or 'url' in file else None
                        image_id = file.get('id') if file_id is None else None
                        
                        # Create evaluation for this file and prompt
                        evaluation_create = crud.EvaluationCreate(
                            file_id=file_id if file_id else None,
                            image_id=image_id if not file_id else None,
                            evaluation_run_id=run_id,
                            prompt_version_id=prompt_config['prompt_version_id'],
                            force_reprocess=True
                        )
                        
                        # Create the evaluation record
                        db_evaluation = await crud.create_evaluation(db, evaluation_create)
                        
                        # Update progress
                        processed_count += 1
                        progress_percentage = int((processed_count / (total_files * len(prompt_configs))) * 100)
                        
                        db.table('evaluation_runs').update({
                            'progress_percentage': progress_percentage,
                            'current_step': f"Processing file {file.get('number', file.get('id'))} with {prompt_config['label']}"
                        }).eq('id', run_id).execute()
                        
                        # Fetch the prompt_text for this prompt_version
                        prompt_text = None
                        prompt_version_str = prompt_config['version']
                        prompt_family_id = prompt_config.get('family_id')
                        if prompt_version_str and prompt_family_id is not None:
                            prompt_version_obj = await crud.get_prompt_version_by_version_string(db, prompt_version_str, prompt_family_id)
                            if prompt_version_obj:
                                prompt_text = prompt_version_obj.get('prompt_text')
                        logging.info(f"[EVAL-RUN {run_id}] Using prompt_version: {prompt_version_str} for file {file.get('number', file.get('id'))}")
                        if prompt_text:
                            logging.info(f"[EVAL-RUN {run_id}] prompt_text (first 200 chars): {prompt_text[:200]}")
                        else:
                            logging.warning(f"[EVAL-RUN {run_id}] No prompt_text found for prompt_version: {prompt_version_str}")

                        # Get dataset information to determine input_type for evaluation runs
                        dataset_input_type = None
                        if file_id:
                            # Get the dataset through the file association
                            file_result = db.table('dataset_files').select('*, dataset_file_associations(dataset_id)').eq('id', file_id).execute()
                            if file_result.data and file_result.data[0]:
                                file_data = file_result.data[0]
                                associations = file_data.get('dataset_file_associations', [])
                                if associations:
                                    dataset_id = associations[0].get('dataset_id')
                                    if dataset_id:
                                        dataset_result = db.table('datasets').select('input_type').eq('id', dataset_id).execute()
                                        if dataset_result.data and dataset_result.data[0]:
                                            dataset_input_type = dataset_result.data[0].get('input_type')
                                            logging.info(f"[EVAL-RUN {run_id}] Dataset input_type: {dataset_input_type}")

                        # Process the evaluation
                        start_time = time.time()
                        result = await orchestrator.process_single_evaluation(
                            file.get('url'),
                            file.get('human_evaluation_text', file.get('expected_output', '')),
                            file.get('number', file.get('id')),
                            prompt_text,
                            file_id,  # Pass the file_id to the orchestrator
                            dataset_input_type  # Pass the dataset input_type to the orchestrator
                        )
                        end_time = time.time()
                        latency_ms = int((end_time - start_time) * 1000)
                        
                        if result.get('success'):
                            # Update evaluation with results
                            word_evaluations = []
                            evaluation_data = result.get('evaluation', {})
                            
                            for word_eval in evaluation_data.get('word_evaluations', []):
                                word_evaluations.append(crud.WordEvaluationCreate(
                                    reference_word=word_eval.get('reference_word', ''),
                                    transcribed_word=word_eval.get('transcribed_word'),
                                    match=word_eval.get('match', False),
                                    reason_diff=word_eval.get('reason_diff', ''),
                                    word_position=word_eval.get('word_position', 0)
                                ))
                            
                            update_data = crud.EvaluationUpdate(
                                ocr_output=evaluation_data.get('ocr_output', ''),
                                accuracy=evaluation_data.get('accuracy', 0),
                                correct_words=evaluation_data.get('correct_words', 0),
                                total_words=evaluation_data.get('total_words', 0),
                                processing_status="success",
                                progress_percentage=100,
                                current_step="Completed",
                                word_evaluations=word_evaluations
                            )
                            
                            await crud.update_evaluation(db, db_evaluation['id'], update_data)

                            # Add API log entry
                            try:
                                log_entry = crud.APILogCreate(
                                    image_url=file.get('url'),
                                    ocr_output=evaluation_data.get('ocr_output', ''),
                                    prompt_version=prompt_config['version'],
                                    user_id=evaluation_run['user_id'],
                                    log_metadata={
                                        "tokens_used": result.get('tokens_used', 0),  # Placeholder, orchestrator needs to return this
                                        "latency_ms": latency_ms
                                    }
                                )
                                await crud.create_api_log(db, log_entry)
                            except Exception as e:
                                logging.error(f"Failed to create API log for evaluation {db_evaluation['id']} in run {run_id}: {str(e)}")
                        else:
                            # Update with error
                            error_msg = result.get('error', 'Unknown error')
                            logging.error(f"OCR processing failed for file {file.get('number', file.get('id'))}: {error_msg}")
                            await crud.update_evaluation(
                                db,
                                db_evaluation['id'],
                                crud.EvaluationUpdate(
                                    processing_status="failed",
                                    progress_percentage=0,
                                    current_step="Failed",
                                    error_message=error_msg
                                )
                            )
                    
                    except Exception as e:
                        # Log error but continue with other evaluations
                        error_msg = f"Error processing file {file.get('id')} with prompt {prompt_config['label']}: {str(e)}"
                        logging.error(error_msg)
                        print(error_msg)
                        
                        # Update evaluation with error if it was created
                        if 'db_evaluation' in locals():
                            await crud.update_evaluation(
                                db,
                                db_evaluation['id'],
                                crud.EvaluationUpdate(
                                    processing_status="failed",
                                    progress_percentage=0,
                                    current_step="Failed",
                                    error_message=str(e)
                                )
                            )
                        continue
            
            # Mark run as completed
            db.table('evaluation_runs').update({
                'status': 'success',
                'progress_percentage': 100,
                'current_step': 'Evaluation run completed',
                'completed_at': datetime.utcnow().isoformat()
            }).eq('id', run_id).execute()
            
        except Exception as e:
            # Update run with error
            db.table('evaluation_runs').update({
                'status': 'failed',
                'current_step': f'Failed: {str(e)}'
            }).eq('id', run_id).execute()
            logging.error(f"Error in evaluation run {run_id}: {str(e)}")
            raise

# Dataset endpoints
@router.get("/api/datasets", response_model=List[Dataset])
async def get_datasets(
    user_id: str,
    db: Client = Depends(get_db)
    ):
    """Get all evaluation datasets for a user"""
    return await crud.get_datasets(db, user_id=user_id)

@router.get("/api/datasets/{dataset_id}", response_model=DatasetWithFiles)
async def get_dataset(dataset_id: int, db: Client = Depends(get_db)):
    """Get a specific dataset with its files"""
    dataset = await crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.post("/api/datasets", response_model=Dataset)
async def create_dataset(dataset: DatasetCreate, db: Client = Depends(get_db)):
    """Create a new dataset"""
    logging.info(f"Creating dataset: {dataset.dict()}")
    try:
        result = await crud.create_dataset(db, dataset)
        logging.info(f"Dataset created successfully: {result}")
        
        # Ensure image_count is not None
        if result and result.get('image_count') is None:
            result['image_count'] = 0
        
        return result
    except Exception as e:
        logging.error(f"Error creating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

@router.put("/api/datasets/{dataset_id}", response_model=Dataset)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: Client = Depends(get_db)
):
    """Update a dataset"""
    dataset = await crud.update_dataset(db, dataset_id, dataset_update)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Client = Depends(get_db)):
    """Delete a dataset and all its associations"""
    try:
        # Delete dataset (cascade will handle associations)
        result = db.table('datasets').delete().eq('id', dataset_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    except Exception as e:
        logging.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/evaluation-runs/{run_id}/cancel")
async def cancel_evaluation_run(run_id: int, db: Client = Depends(get_db)):
    """Cancel a running evaluation"""
    try:
        db.table('evaluation_runs')\
            .update({"status": "failed", "current_step": "Cancelled by user"})\
            .eq('id', run_id)\
            .execute()
        return {"success": True, "run_id": run_id, "status": "cancelled"}
    except Exception as e:
        logging.error(f"Error cancelling run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/evaluation-runs/{run_id}/link-evaluations-simple")
async def link_evaluations_simple(
    run_id: int,
    db: Client = Depends(get_db)
):
    """Simple endpoint to link evaluations without triggering re-processing"""
    try:
        # Get all evaluations that match criteria and update them one by one
        # This avoids the long-running query issue

        # First, get evaluations to update
        evals_result = db.table('evaluations')\
            .select('id')\
            .eq('prompt_version_id', 46)\
            .gte('created_at', '2025-11-03')\
            .is_('evaluation_run_id', 'null')\
            .execute()

        eval_ids = [e['id'] for e in evals_result.data] if evals_result.data else []

        # Update in small batches
        batch_size = 50
        updated_count = 0

        for i in range(0, len(eval_ids), batch_size):
            batch_ids = eval_ids[i:i+batch_size]
            for eval_id in batch_ids:
                db.table('evaluations')\
                    .update({"evaluation_run_id": run_id})\
                    .eq('id', eval_id)\
                    .execute()
                updated_count += 1

        # Update the run status
        db.table('evaluation_runs')\
            .update({
                "status": "completed",
                "progress_percentage": 100,
                "completed_at": datetime.utcnow().isoformat()
            })\
            .eq('id', run_id)\
            .execute()

        return {
            "success": True,
            "run_id": run_id,
            "evaluations_linked": updated_count,
            "status": "completed"
        }
    except Exception as e:
        logging.error(f"Error linking evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/datasets/{dataset_id}/upload", response_model=Dataset)
async def upload_dataset_files(
    dataset_id: int,
    images_zip: UploadFile = File(...),
    reference_csv: UploadFile = File(...),
    db: Client = Depends(get_db)
):
    """Upload images and reference CSV for a dataset"""
    dataset = await crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate files
    if not images_zip.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Images file must be a ZIP archive")
    if not reference_csv.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Reference file must be CSV")
    
    # Process the uploaded files
    result = await crud.process_dataset_upload(db, dataset_id, images_zip, reference_csv)
    return result

@router.delete("/api/datasets/{dataset_id}/images/{image_id}")
async def delete_file_from_dataset(dataset_id: int, image_id: int, db: Client = Depends(get_db)):
    """Delete a file from a dataset, including the association and the file itself."""
    success = await crud.delete_file_from_dataset(db, dataset_id, image_id)
    if not success:
        raise HTTPException(status_code=404, detail="File or dataset not found")
    return {"message": "File deleted from dataset and removed from database"}

@router.put("/api/datasets/{dataset_id}/images/{image_id}", response_model=DatasetFile)
async def update_file_in_dataset(
    dataset_id: int,
    image_id: int,
    image_update: ImageUpdate,
    db: Client = Depends(get_db)
):
    """Update reference_text for a file in a dataset (no human_evaluation_text)"""
    # Check association
    is_associated = await crud.is_file_in_dataset(db, image_id, dataset_id)
    if not is_associated:
        raise HTTPException(status_code=404, detail="File not found in dataset")
    # Update file (only expected_output)
    file = await crud.update_dataset_file(db, image_id, image_update)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

# Prompt Family endpoints
@router.get("/api/prompt-families", response_model=List[PromptFamily])
async def get_prompt_families(user_id: str, db: Client = Depends(get_db)):
    """Get all prompt families for a user"""
    return await crud.get_prompt_families(db, user_id=user_id)

@router.get("/api/prompt-families/{family_id}", response_model=PromptFamilyWithVersions)
async def get_prompt_family(family_id: int, db: Client = Depends(get_db)):
    """Get a specific prompt family with its versions"""
    family = await crud.get_prompt_family(db, family_id)
    if not family:
        raise HTTPException(status_code=404, detail="Prompt family not found")
    return family

@router.put("/api/prompt-families/{family_id}", response_model=PromptFamily)
async def update_prompt_family(
    family_id: int,
    family_update: PromptFamilyCreate,
    db: Client = Depends(get_db)
):
    """Update a prompt family"""
    # First check if the family exists
    existing_family = await crud.get_prompt_family(db, family_id)
    if not existing_family:
        raise HTTPException(status_code=404, detail="Prompt family not found")
    
    # Update the family
    updated_family = await crud.update_prompt_family(db, family_id, family_update)
    return updated_family

@router.post("/api/prompt-families", response_model=PromptFamily)
async def create_prompt_family(family: PromptFamilyCreate, db: Client = Depends(get_db)):
    """Create a new prompt family"""
    return await crud.create_prompt_family(db, family)

# Prompt Version endpoints
@router.get("/api/prompt-families/{family_id}/versions", response_model=List[PromptVersion])
async def get_prompt_versions(family_id: int, user_id: str, db: Client = Depends(get_db)):
    """Get all versions for a prompt family"""
    versions = await crud.get_prompt_versions(db, family_id, user_id)
    result = []
    for v in versions:
        # Parse issues - v is a dictionary, so use dictionary access
        if isinstance(v.get('issues'), str):
            try:
                issues = json.loads(v['issues'])
            except Exception:
                issues = []
        elif isinstance(v.get('issues'), list):
            issues = v['issues']
        else:
            issues = []
        
        # Create PromptVersion object from dictionary
        pv = PromptVersion(
            id=v.get('id'),
            family_id=v.get('family_id'),
            version=v.get('version'),
            prompt_text=v.get('prompt_text'),
            changelog_message=v.get('changelog_message'),
            status=v.get('status'),
            author=v.get('author'),
            created_at=v.get('created_at'),
            last_evaluation_accuracy=v.get('last_evaluation_accuracy'),
            user_id=v.get('user_id'),
            issues=issues
        )
        result.append(pv)
    return result

@router.post("/api/prompt-families/{family_id}/versions", response_model=PromptVersion)
async def create_prompt_version(
    family_id: int,
    version: PromptVersionCreate,
    db: Client = Depends(get_db)
):
    """Create a new version of a prompt"""
    # Validate family exists
    family = await crud.get_prompt_family(db, family_id)
    if not family:
        raise HTTPException(status_code=404, detail="Prompt family not found")
    
    # Generate version number based on type
    next_version = await crud.generate_next_version(db, family_id, version.version_type)
    
    # Set the generated version number
    version.version = next_version
    
    # Create the version
    return await crud.create_prompt_version(db, version)

@router.put("/api/prompt-versions/{version_id}", response_model=PromptVersion)
async def update_prompt_version(
    version_id: int,
    version_update: PromptVersionUpdate,
    db: Client = Depends(get_db)
):
    """Update a prompt version"""
    # First check if the version exists
    existing_version = await crud.get_prompt_version(db, version_id)
    if not existing_version:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    
    # Update the version
    updated_version = await crud.update_prompt_version(db, version_id, version_update)
    if not updated_version:
        raise HTTPException(status_code=500, detail="Failed to update prompt version")
    
    return updated_version

@router.post("/api/prompt-versions/{version_id}/promote")
async def promote_prompt_version(version_id: int, db: Client = Depends(get_db)):
    """Promote a prompt version to production"""
    result = await crud.promote_prompt_version(db, version_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return {"message": "Prompt version promoted to production"}

@router.patch("/api/prompt-versions/by-version/{version}/issues", response_model=PromptVersion)
async def patch_prompt_version_issues_by_version(
    version: str,
    family_id: int = Query(..., description="Prompt family ID"),
    issue_data: Dict[str, Any] = None,
    db: Client = Depends(get_db)
):
    """Append a new issue instance to the issues field of a prompt version, using the version string and family_id"""
    existing_version = await crud.get_prompt_version_by_version_string(db, version, family_id)
    if not existing_version:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    version_id = existing_version['id']
    # Get current issues (parse from JSON if needed)
    current_issues = existing_version.get('issues', [])
    if isinstance(current_issues, str):
        import json
        try:
            current_issues = json.loads(current_issues)
        except Exception:
            current_issues = []
    # Append the new issue
    if issue_data and 'image_id' in issue_data and 'issue' in issue_data:
        current_issues.append({'image_id': issue_data['image_id'], 'issue': issue_data['issue']})
    else:
        raise HTTPException(status_code=400, detail="Missing image_id or issue in request body")
    version_update = PromptVersionUpdate(issues=current_issues)
    updated_version = await crud.update_prompt_version(db, version_id, version_update)
    if not updated_version:
        raise HTTPException(status_code=500, detail="Failed to update prompt version issues")
    return updated_version

@router.patch("/api/word-evaluations/{word_evaluation_id}/comment")
async def patch_word_evaluation_comment(
    word_evaluation_id: int,
    data: Dict[str, str],
    db: Client = Depends(get_db)
):
    """Update the comments field for a word evaluation and update both the parent evaluation's word_evaluations_json and prompt version issues."""
    # Fetch the word evaluation
    result = db.table('word_evaluations').select('*').eq('id', word_evaluation_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Word evaluation not found")
    word_eval = result.data[0]
    
    # Get and validate the comment
    comment = data.get('comments', '').strip()
    
    # Skip if comment looks like an error message or log
    if any(error_indicator in comment.lower() for error_indicator in [
        'error:', 'exception', 'traceback', 'http request:', 'info:', 
        'debug:', 'warning:', 'critical:', 'fatal:'
    ]):
        return {"message": "Invalid comment format - looks like an error message or log"}
    
    # Update the comments field
    db.table('word_evaluations').update({'comments': comment}).eq('id', word_evaluation_id).execute()
    
    # Fetch all word evaluations for the parent evaluation
    eval_id = word_eval['evaluation_id']
    all_word_evals = db.table('word_evaluations').select('*').eq('evaluation_id', eval_id).order('word_position').execute().data
    
    # Get the evaluation to find prompt_version_id
    evaluation = db.table('evaluations').select('*').eq('id', eval_id).single().execute()
    if not evaluation.data:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    prompt_version_id = evaluation.data.get('prompt_version_id')
    if prompt_version_id:
        # Fetch current prompt version
        prompt_version = db.table('prompt_versions').select('*').eq('id', prompt_version_id).single().execute()
        if prompt_version.data:
            # Parse existing issues or initialize empty list
            current_issues = []
            if prompt_version.data.get('issues'):
                try:
                    current_issues = json.loads(prompt_version.data['issues'])
                except:
                    current_issues = []
            
            # Create new issue entry
            if comment:  # Only add issue if there's a comment
                new_issue = {
                    'id': str(uuid.uuid4()),
                    'word_evaluation_id': word_evaluation_id,
                    'comment': comment,
                    'created_at': datetime.now().isoformat(),
                    'word_position': word_eval.get('word_position'),
                    'word': word_eval.get('transcribed_word'),
                    'type': 'word_evaluation_comment'
                }
                
                # Remove any existing issues for this word evaluation
                current_issues = [issue for issue in current_issues 
                                if not (issue.get('type') == 'word_evaluation_comment' 
                                      and issue.get('word_evaluation_id') == word_evaluation_id)]
                
                # Add new issue
                current_issues.append(new_issue)
            else:
                # If comment is empty, remove any existing issues for this word evaluation
                current_issues = [issue for issue in current_issues 
                                if not (issue.get('type') == 'word_evaluation_comment' 
                                      and issue.get('word_evaluation_id') == word_evaluation_id)]
            
            # Update prompt version with new issues
            db.table('prompt_versions').update({
                'issues': json.dumps(current_issues)
            }).eq('id', prompt_version_id).execute()
    
    # Update the parent evaluation's word_evaluations_json
    db.table('evaluations').update({
        'word_evaluations_json': json.dumps(all_word_evals)
    }).eq('id', eval_id).execute()
    
    return {"message": "Comment updated and issues synced"}

# Evaluation Run endpoints
@router.get("/api/evaluation-runs", response_model=List[EvaluationRunSchema])
async def get_evaluation_runs(user_id: str, db: Client = Depends(get_db)):
    """Get all evaluation runs for a user"""
    return await crud.get_evaluation_runs(db, user_id=user_id)

@router.get("/api/evaluation-runs/{run_id}", response_model=EvaluationRunWithDetails)
async def get_evaluation_run(run_id: int, db: Client = Depends(get_db)):
    """Get a specific evaluation run with full details"""
    run = await crud.get_evaluation_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run

@router.post("/api/evaluation-runs", response_model=EvaluationRunSchema)
async def create_evaluation_run(
    run: EvaluationRunCreate,
    background_tasks: BackgroundTasks,
    db: Client = Depends(get_db)
):
    """Create and start a new evaluation run (A/B test)"""
    logging.info("[API] Entered create_evaluation_run endpoint")
    try:
        # Validate datasets exist
        for dataset_id in run.dataset_ids:
            logging.info(f"[API] Validating dataset_id: {dataset_id}")
            dataset = await crud.get_dataset(db, dataset_id)
            if not dataset:
                logging.error(f"[API] Dataset {dataset_id} not found")
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            if dataset.get('status') != DatasetStatus.VALIDATED:
                logging.error(f"[API] Dataset {dataset.get('name')} is not validated (status: {dataset.get('status')})")
                raise HTTPException(status_code=400, detail=f"Dataset {dataset.get('name')} is not validated")
        
        # Create the evaluation run
        logging.info("[API] Creating evaluation run in DB")
        db_run = await crud.create_evaluation_run(db, run)
        logging.info(f"[API] Evaluation run created with id: {db_run.get('id')}")
        
        # Queue background processing
        logging.info(f"[API] Adding background task for run id: {db_run.get('id')}")
        background_tasks.add_task(process_evaluation_run_background, db_run.get('id'))
        
        logging.info(f"[API] Returning evaluation run with id: {db_run.get('id')}")
        # return db_run
        return EvaluationRunSchema(
            id = db_run.get('id'),
            name = db_run.get('name'),
            description = db_run.get('description'),
            hypothesis = db_run.get('hypothesis'),
            status = db_run.get('status'),
            progress_percentage = db_run.get('progress_percentage'),
            current_step = db_run.get('current_step'),
            created_at = db_run.get('created_at'),
            updated_at = db_run.get('updated_at'),
            completed_at = db_run.get('completed_at'),
            dataset_ids=[d.get('id') for d in db_run.get('datasets', [])],
            user_id=db_run.get('user_id')
        )
    except Exception as e:
        logging.exception(f"[API] Exception in create_evaluation_run: {str(e)}")
        raise

@router.get("/api/evaluation-runs/{run_id}/comparison", response_model=ComparisonResults)
async def get_evaluation_comparison(run_id: int, db: Client = Depends(get_db)):
    """Get detailed comparison results for an evaluation run"""
    comparison = await crud.get_evaluation_comparison(db, run_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="Evaluation run not found or not completed")
    return comparison

# Real-time WebSocket endpoint for live progress
@router.websocket("/ws/evaluation-runs/{run_id}/progress")
async def websocket_evaluation_progress(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time evaluation progress"""
    await websocket.accept()
    
    try:
        while True:
            # Get current progress
            db = next(get_db())
            progress = await crud.get_evaluation_run_progress(db, run_id)
            if progress:
                await websocket.send_json(progress)
                
                # If completed, send final update and close
                if progress and progress.get('status') in ['success', 'failed']:
                    break
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        pass

# Historical Analysis endpoints
@router.get("/api/analysis/performance-trends", response_model=List[PerformanceTrend])
async def get_performance_trends(
    prompt_family_id: Optional[int] = Query(None),
    dataset_id: Optional[int] = Query(None),
    days_back: int = Query(30),
    db: Client = Depends(get_db)
):
    """Get performance trends over time"""
    return await crud.get_performance_trends(db, prompt_family_id, dataset_id, days_back)

@router.get("/api/analysis/regression-alerts")
async def get_regression_alerts(db: Client = Depends(get_db)):
    """Get active regression alerts"""
    return await crud.get_regression_alerts(db)

# API Log endpoint
@router.get("/api/api-logs", response_model=List[APILog])
async def get_api_logs(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Client = Depends(get_db)
):
    """Get API logs for the current user"""
    return await crud.get_api_logs_for_user(db, user_id=user_id, skip=skip, limit=limit)

# API Key Management endpoints
@router.get("/api/api-keys", response_model=List[APIKey])
async def get_api_keys(db: Client = Depends(get_db)):
    """Get all API keys for the current user"""
    return await crud.get_api_keys(db)

@router.post("/api/api-keys", response_model=APIKey)
async def create_api_key(key_data: APIKeyCreate, db: Client = Depends(get_db)):
    """Create a new API key"""
    return await crud.create_api_key(db, key_data)

@router.delete("/api/api-keys/{key_id}")
async def revoke_api_key(key_id: int, db: Client = Depends(get_db)):
    """Revoke an API key"""
    success = await crud.revoke_api_key(db, key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"message": "API key revoked"}

@router.get("/api/api-keys/{key_id}/usage", response_model=APIUsageStats)
async def get_api_key_usage(key_id: int, db: Client = Depends(get_db)):
    """Get usage statistics for an API key"""
    stats = await crud.get_api_key_usage(db, key_id)
    if not stats:
        raise HTTPException(status_code=404, detail="API key not found")
    return stats

@router.get("/api/debug/dataset-images-table")
async def debug_dataset_images_table_endpoint(db: Client = Depends(get_db)):
    """Debug endpoint to check dataset_images table structure"""
    from . import crud
    return await crud.debug_dataset_images_table(db)

@router.post("/api/deploy/infer", response_model=DeployInferResponse)
async def deploy_infer(
    request: GenericEvalRequest,
    db: Client = Depends(get_db)
):
    try:
        eval_id = request.eval_config_id
        logging.info(f"Starting deploy inference for eval_id: {eval_id}")

        # Get evaluation record with prompt_version_id, file_id, and evaluation_run_id
        eval_result = db.table('evaluations').select('prompt_version_id, file_id, evaluation_run_id').eq('id', eval_id).execute()
        if not eval_result.data:
            raise HTTPException(status_code=404, detail=f"Evaluation {eval_id} not found")

        evaluation = eval_result.data[0]
        prompt_version_id = evaluation.get('prompt_version_id')
        file_id = evaluation.get('file_id')
        evaluation_run_id = evaluation.get('evaluation_run_id')

        if not prompt_version_id:
            raise HTTPException(status_code=400, detail=f"Evaluation {eval_id} has no prompt_version_id")
        if not file_id:
            raise HTTPException(status_code=400, detail=f"Evaluation {eval_id} has no file_id")

        # Get prompt_text from prompt_versions table
        prompt_result = db.table('prompt_versions').select('prompt_text').eq('id', prompt_version_id).execute()
        if not prompt_result.data:
            raise HTTPException(status_code=404, detail=f"Prompt version {prompt_version_id} not found")
        prompt_text = prompt_result.data[0].get('prompt_text')
        if not prompt_text:
            raise HTTPException(status_code=400, detail=f"Prompt version {prompt_version_id} has no prompt_text")

        # Get dataset information to determine output_type/input_type
        dataset_output_type = "text"  # Default fallback
        dataset_input_type = None
        try:
            dataset_result = db.table('dataset_file_associations').select('dataset_id').eq('file_id', file_id).execute()
            if dataset_result.data:
                dataset_id = dataset_result.data[0].get('dataset_id')
                if dataset_id:
                    dataset_info_result = db.table('datasets').select('output_type', 'input_type').eq('id', dataset_id).execute()
                    if dataset_info_result.data:
                        dataset_output_type = dataset_info_result.data[0].get('output_type', 'text')
                        dataset_input_type = dataset_info_result.data[0].get('input_type', None)
                        logging.info(f"Found dataset output_type: {dataset_output_type}, input_type: {dataset_input_type}")
        except Exception as e:
            logging.warning(f"Could not get dataset output_type/input_type for file_id {file_id}: {str(e)}")

        orchestrator = get_ocr_orchestrator()

        # If input is provided, use its url and reference_text as needed
        if request.input and isinstance(request.input, dict) and "url" in request.input:
            file_url = request.input.get("url")
            # Use reference_text only if output_type or input_type is 'image'
            expected_output = None
            if (dataset_input_type and str(dataset_input_type).lower() == "image"):
                expected_output = request.input.get("reference_text")
            # Call orchestrator
            result = await orchestrator.process_single_evaluation(
                file_url,
                expected_output,
                "input",  # Use a dummy file_id or identifier
                prompt_text,
                None,  # file_id
                (dataset_input_type or "IMAGE").upper()
            )

            if (dataset_input_type and str(dataset_input_type).lower() == "image"):
                # Compose output as required for images
                if result.get('success'):
                    evaluation_data = result.get('evaluation', {})
                    output = {
                        "full_text": evaluation_data.get('ocr_output', ''),
                        "word_evaluations": evaluation_data.get('word_evaluations', []),
                        "total_words": evaluation_data.get('total_words', 0),
                        "correct_words": evaluation_data.get('correct_words', 0),
                        "accuracy": evaluation_data.get('accuracy', 0),
                    }

                    # Save evaluation to database
                    try:
                        new_eval = {
                            'prompt_version_id': prompt_version_id,
                            'evaluation_run_id': evaluation_run_id,  # Link to same run as source
                            'ocr_output': evaluation_data.get('ocr_output', ''),
                            'accuracy': evaluation_data.get('accuracy', 0),
                            'correct_words': evaluation_data.get('correct_words', 0),
                            'total_words': evaluation_data.get('total_words', 0),
                            'processing_status': 'success',
                            'latency_ms': result.get('tokens_used', 0),  # Store tokens in latency_ms temporarily
                            'model_info': {'tokens_used': result.get('tokens_used', 0)}
                        }

                        # Try to create a dataset_file record if URL is provided
                        if file_url and expected_output:
                            file_create_result = db.table('dataset_files').insert({
                                'url': file_url,
                                'expected_output': expected_output,
                                'file_type': 'image',
                                'number': f'deploy_{eval_id}_{int(time.time())}',
                                'user_id': 'anonymous'  # Default user for API-created files
                            }).execute()

                            if file_create_result.data:
                                new_eval['file_id'] = file_create_result.data[0]['id']

                        eval_insert_result = db.table('evaluations').insert(new_eval).execute()

                        if eval_insert_result.data and evaluation_data.get('word_evaluations'):
                            new_eval_id = eval_insert_result.data[0]['id']
                            # Save word evaluations
                            word_evals_to_insert = []
                            for we in evaluation_data.get('word_evaluations', []):
                                word_evals_to_insert.append({
                                    'evaluation_id': new_eval_id,
                                    'reference_word': we.get('reference_word', ''),
                                    'transcribed_word': we.get('transcribed_word', ''),
                                    'match': we.get('match', False),
                                    'reason_diff': we.get('reason_diff', ''),
                                    'word_position': we.get('word_position', 0)
                                })

                            if word_evals_to_insert:
                                db.table('word_evaluations').insert(word_evals_to_insert).execute()

                            logging.info(f"Saved evaluation {new_eval_id} to database with {len(word_evals_to_insert)} word evaluations")
                    except Exception as db_error:
                        logging.error(f"Failed to save evaluation to database: {str(db_error)}")
                        # Don't fail the request if DB save fails

                    return DeployInferResponse(
                        output=output,
                        tokens_used=result.get('tokens_used', 0),
                        success=True,
                        error_message=None
                    )
                else:
                    return DeployInferResponse(
                        output={"full_text": "", "word_evaluations": [], "total_words": 0, "correct_words": 0, "accuracy": 0},
                        tokens_used=result.get('tokens_used', 0),
                        success=False,
                        error_message=result.get('error', 'Unknown error')
                    )
            else:
                # Fallback to original output logic for non-image types
                if result.get('success'):
                    evaluation_data = result.get('evaluation', {})
                    raw_output = evaluation_data.get('ocr_output', '')
                    formatted_output = raw_output
                    if dataset_output_type == "json":
                        try:
                            if isinstance(raw_output, str):
                                import json
                                formatted_output = json.loads(raw_output)
                            else:
                                formatted_output = raw_output
                        except (json.JSONDecodeError, TypeError):
                            formatted_output = raw_output
                    else:
                        formatted_output = str(raw_output) if raw_output is not None else ""
                    return DeployInferResponse(
                        output=formatted_output,
                        tokens_used=result.get('tokens_used', 0),
                        success=True,
                        error_message=None
                    )
                else:
                    return DeployInferResponse(
                        output="" if dataset_output_type == "text" else {},
                        tokens_used=result.get('tokens_used', 0),
                        success=False,
                        error_message=result.get('error', 'Unknown error')
                    )

        # Fallback to current DB-based logic if no input is provided
        # Get file_type and file_url from DB
        file_result = db.table('dataset_files').select('file_type, url').eq('id', file_id).execute()
        if not file_result.data:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        file_data = file_result.data[0]
        file_type = file_data.get('file_type', 'image')
        file_url = file_data.get('url')
        if not file_url:
            raise HTTPException(status_code=400, detail=f"File {file_id} has no URL")

        logging.info(f"Processing deploy_infer eval_id: {eval_id}, file_type: {file_type}, prompt_version_id: {prompt_version_id}, output_type: {dataset_output_type}")

        # Process the evaluation using orchestrator (no expected_output)
        start_time = time.time()
        result = await orchestrator.process_single_evaluation(
            file_url,
            None,  # No expected_output
            str(file_id),
            prompt_text,
            file_id,
            file_type.upper() if file_type else None
        )
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        if result.get('success'):
            evaluation_data = result.get('evaluation', {})
            if (file_type and str(file_type).lower() == "image"):
                output = {
                    "full_text": evaluation_data.get('ocr_output', ''),
                    "word_evaluations": evaluation_data.get('word_evaluations', []),
                    "total_words": evaluation_data.get('total_words', 0),
                    "correct_words": evaluation_data.get('correct_words', 0),
                    "accuracy": evaluation_data.get('accuracy', 0),
                }
                return DeployInferResponse(
                    output=output,
                    tokens_used=result.get('tokens_used', 0),
                    success=True,
                    error_message=None
                )
            raw_output = evaluation_data.get('ocr_output', '')
            formatted_output = raw_output
            if dataset_output_type == "json":
                try:
                    if isinstance(raw_output, str):
                        import json
                        formatted_output = json.loads(raw_output)
                    else:
                        formatted_output = raw_output
                except (json.JSONDecodeError, TypeError):
                    formatted_output = raw_output
            else:
                formatted_output = str(raw_output) if raw_output is not None else ""
            return DeployInferResponse(
                output=formatted_output,
                tokens_used=result.get('tokens_used', 0),
                success=True,
                error_message=None
            )
        else:
            if (file_type and str(file_type).lower() == "image"):
                return DeployInferResponse(
                    output={"full_text": "", "word_evaluations": [], "total_words": 0, "correct_words": 0, "accuracy": 0},
                    tokens_used=result.get('tokens_used', 0),
                    success=False,
                    error_message=result.get('error', 'Unknown error')
                )
            return DeployInferResponse(
                output="" if dataset_output_type == "text" else {},
                tokens_used=result.get('tokens_used', 0),
                success=False,
                error_message=result.get('error', 'Unknown error')
            )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in deploy_infer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deploy inference failed: {str(e)}")

@router.get("/api/eval-viewer")
async def get_eval_viewer_data(
    eval_id: Optional[int] = Query(None, description="Specific evaluation ID to view"),
    evaluation_run_id: Optional[int] = Query(None, description="View all evaluations from a run"),
    prompt_version_id: Optional[int] = Query(None, description="Filter by prompt version"),
    file_id: Optional[int] = Query(None, description="Filter by file ID"),
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    db: Client = Depends(get_db)
):
    """
    Get evaluation data for telemetry and debugging.
    Shows input/output of all API calls in an accessible manner.
    """
    try:
        # If specific eval_id is provided, return that evaluation with full details
        if eval_id:
            result = db.table('evaluations').select('*').eq('id', eval_id).execute()

            if not result.data:
                raise HTTPException(status_code=404, detail=f"Evaluation {eval_id} not found")

            evaluation = result.data[0]

            # Fetch related data separately
            # Fetch file info
            if evaluation.get('file_id'):
                file_result = db.table('dataset_files').select('*').eq('id', evaluation['file_id']).execute()
                if file_result.data:
                    evaluation['file'] = file_result.data[0]
                    # Fetch dataset info
                    if evaluation['file'].get('dataset_id'):
                        dataset_result = db.table('datasets').select('*').eq('id', evaluation['file']['dataset_id']).execute()
                        if dataset_result.data:
                            evaluation['file']['dataset'] = dataset_result.data[0]

            # Fetch word evaluations
            word_evals_result = db.table('word_evaluations').select('*').eq('evaluation_id', eval_id).order('word_position').execute()
            evaluation['word_evaluations'] = word_evals_result.data or []

            # Fetch prompt version
            if evaluation.get('prompt_version_id'):
                prompt_result = db.table('prompt_versions').select('*').eq('id', evaluation['prompt_version_id']).execute()
                if prompt_result.data:
                    evaluation['prompt_version'] = prompt_result.data[0]

            # Fetch API logs for this evaluation if available
            # Note: api_logs table may not have evaluation_id column, so we try to fetch by matching
            api_logs = []
            try:
                # Try to get logs by user_id and around the same time
                if evaluation.get('file') and evaluation['file'].get('url'):
                    api_logs_result = db.table('api_logs').select('*').eq('image_url', evaluation['file']['url']).order('created_at', desc=True).limit(10).execute()
                    api_logs = api_logs_result.data if api_logs_result.data else []
            except Exception as e:
                logging.warning(f"Could not fetch api_logs for evaluation {eval_id}: {str(e)}")
                api_logs = []

            return {
                "evaluation": evaluation,
                "api_logs": api_logs
            }

        # Otherwise, return list of evaluations based on filters
        query = db.table('evaluations').select('*')

        if evaluation_run_id:
            query = query.eq('evaluation_run_id', evaluation_run_id)

        if prompt_version_id:
            query = query.eq('prompt_version_id', prompt_version_id)

        if file_id:
            query = query.eq('file_id', file_id)

        # If dataset_id is provided, we need to join through files
        if dataset_id:
            # First get all file_ids for this dataset
            files_result = db.table('dataset_files').select('id').eq('dataset_id', dataset_id).execute()
            if files_result.data:
                file_ids = [f['id'] for f in files_result.data]
                query = query.in_('file_id', file_ids)

        # Order by most recent first
        # If viewing a specific run, don't limit; otherwise limit to 100
        if not evaluation_run_id:
            query = query.limit(100)
        query = query.order('created_at', desc=True)

        result = query.execute()
        evaluations = result.data or []

        # Enrich evaluations with related data using batch queries
        # Collect all unique IDs
        file_ids = list(set([e.get('file_id') for e in evaluations if e.get('file_id')]))
        prompt_version_ids = list(set([e.get('prompt_version_id') for e in evaluations if e.get('prompt_version_id')]))

        # Batch fetch files
        files_map = {}
        if file_ids:
            files_result = db.table('dataset_files').select('*').in_('id', file_ids).execute()
            files_map = {f['id']: f for f in (files_result.data or [])}

        # Batch fetch datasets (collect dataset_ids from files)
        dataset_ids = list(set([f.get('dataset_id') for f in files_map.values() if f.get('dataset_id')]))
        datasets_map = {}
        if dataset_ids:
            datasets_result = db.table('datasets').select('id, name, description').in_('id', dataset_ids).execute()
            datasets_map = {d['id']: d for d in (datasets_result.data or [])}

        # Batch fetch prompt versions
        prompt_versions_map = {}
        if prompt_version_ids:
            prompts_result = db.table('prompt_versions').select('*').in_('id', prompt_version_ids).execute()
            prompt_versions_map = {p['id']: p for p in (prompts_result.data or [])}

        # Attach related data to evaluations
        for evaluation in evaluations:
            # Attach file info
            if evaluation.get('file_id') and evaluation['file_id'] in files_map:
                evaluation['file'] = files_map[evaluation['file_id']]
                # Attach dataset info to file
                if evaluation['file'].get('dataset_id') and evaluation['file']['dataset_id'] in datasets_map:
                    evaluation['file']['dataset'] = datasets_map[evaluation['file']['dataset_id']]

            # Skip word_evaluations in list view for performance (only fetch in detail view)
            evaluation['word_evaluations'] = []

            # Attach prompt version
            if evaluation.get('prompt_version_id') and evaluation['prompt_version_id'] in prompt_versions_map:
                evaluation['prompt_version'] = prompt_versions_map[evaluation['prompt_version_id']]

        return {
            "evaluations": evaluations,
            "total": len(evaluations),
            "filters": {
                "evaluation_run_id": evaluation_run_id,
                "prompt_version_id": prompt_version_id,
                "file_id": file_id,
                "dataset_id": dataset_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in eval-viewer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch evaluation data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000, reload=True) 