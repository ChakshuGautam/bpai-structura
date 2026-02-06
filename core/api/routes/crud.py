from supabase import Client
from typing import List, Optional, Dict, Any
import json
import csv
import os
import zipfile
import tempfile
from datetime import datetime, timedelta
import logging

from ...models.ocr_models import (
    ImageCreate, ImageUpdate, EvaluationCreate, EvaluationUpdate,
    PromptTemplateCreate, PromptTemplateUpdate, WordEvaluationCreate,
    ImageFilter, PaginationParams,
    DatasetCreate, DatasetUpdate, PromptFamilyCreate,
    PromptVersionCreate, PromptVersionUpdate, EvaluationRunCreate,
    VersionType, ProcessingStatus, DatasetStatus, PromptStatus,
    APIKeyCreate, APILogCreate, DatasetFileCreate, DatasetFileUpdate
)

# Image CRUD operations
async def create_image(db: Client, image: ImageCreate) -> Dict[str, Any]:
    """Create a new image using Supabase"""
    image_data = image.dict()
    image_data['created_at'] = datetime.utcnow().isoformat()
    image_data['updated_at'] = datetime.utcnow().isoformat()
    
    result = db.table('images').insert(image_data).execute()
    return result.data[0] if result.data else None

async def get_image(db: Client, image_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific image with its evaluations"""
    result = db.table('images').select('*, evaluations(*)').eq('id', image_id).execute()
    return result.data[0] if result.data else None

async def get_image_by_number(db: Client, number: str) -> Optional[Dict[str, Any]]:
    """Get image by number"""
    result = db.table('images').select('*').eq('number', number).execute()
    return result.data[0] if result.data else None

async def get_images(
    db: Client, 
    filters: ImageFilter = None, 
    pagination: PaginationParams = None
) -> tuple[List[Dict[str, Any]], int]:
    """Get paginated list of images with optional filters"""
    query = db.table('images').select('*, evaluations(*)')
    
    # Apply filters
    if filters:
        if filters.has_evaluations is not None:
            if filters.has_evaluations:
                # Images with evaluations
                query = query.not_.is_('evaluations', 'null')
            else:
                # Images without evaluations
                query = query.is_('evaluations', 'null')
        
        if filters.processing_status:
            query = query.eq('evaluations.processing_status', filters.processing_status)
        
        if filters.prompt_version:
            query = query.eq('evaluations.prompt_version', filters.prompt_version)
        
        if filters.accuracy_min is not None:
            query = query.gte('evaluations.accuracy', filters.accuracy_min)
        
        if filters.accuracy_max is not None:
            query = query.lte('evaluations.accuracy', filters.accuracy_max)
        
        if filters.created_after:
            query = query.gte('created_at', filters.created_after.isoformat())
        
        if filters.created_before:
            query = query.lte('created_at', filters.created_before.isoformat())
    
    # Get total count
    count_result = query.execute()
    total = len(count_result.data) if count_result.data else 0
    
    # Apply pagination
    if pagination:
        query = query.range(pagination.skip, pagination.skip + pagination.limit - 1)
    
    result = query.execute()
    return result.data or [], total

async def update_image(db: Client, image_id: int, image_update: ImageUpdate) -> Optional[Dict[str, Any]]:
    """Update an existing image"""
    update_data = image_update.dict(exclude_unset=True)
    update_data['updated_at'] = datetime.utcnow().isoformat()
    
    result = db.table('images').update(update_data).eq('id', image_id).execute()
    return result.data[0] if result.data else None

async def delete_image(db: Client, image_id: int) -> bool:
    """Delete an image"""
    result = db.table('images').delete().eq('id', image_id).execute()
    return len(result.data) > 0 if result.data else False

async def get_all_images_with_evaluations(db: Client) -> List[Dict[str, Any]]:
    """Get all images with their evaluations for CSV export"""
    result = db.table('images').select('*, evaluations(*)').order('id').execute()
    return result.data or []

# Evaluation CRUD operations
async def create_evaluation(db: Client, evaluation: EvaluationCreate) -> Dict[str, Any]:
    """Create a new evaluation"""
    # Determine which ID to use (file_id takes precedence)
    target_id = evaluation.file_id if evaluation.file_id is not None else evaluation.image_id
    target_field = 'file_id' if evaluation.file_id is not None else 'image_id'
    
    if target_id is None:
        raise ValueError("Either file_id or image_id must be provided")
    
    # Check if evaluation already exists for this file/image and prompt version id
    existing_result = db.table('evaluations').select('*').eq(target_field, target_id).eq('prompt_version_id', evaluation.prompt_version_id).execute()
    existing_eval = existing_result.data[0] if existing_result.data else None
    
    if existing_eval and not evaluation.force_reprocess:
        return existing_eval
    
    evaluation_data = {
        target_field: target_id,
        'evaluation_run_id': evaluation.evaluation_run_id,
        'prompt_version_id': evaluation.prompt_version_id,
        'processing_status': 'pending',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'model_id': getattr(evaluation, 'model_id', None),
        'model_info': getattr(evaluation, 'model_info', None),
    }
    
    result = db.table('evaluations').insert(evaluation_data).execute()
    return result.data[0] if result.data else None

async def get_evaluation(db: Client, evaluation_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific evaluation with full details"""
    # Try to get evaluation with file join first (new approach)
    result = db.table('evaluations').select('*, file:dataset_files(*), word_evaluations(*)').eq('id', evaluation_id).execute()
    evaluation = result.data[0] if result.data else None
    
    if evaluation and evaluation.get('file'):
        return evaluation
    
    # Fallback to image join (backward compatibility)
    result = db.table('evaluations').select('*, image:images(*), word_evaluations(*)').eq('id', evaluation_id).execute()
    return result.data[0] if result.data else None

async def get_evaluations(
    db: Client, 
    image_id: Optional[int] = None,
    file_id: Optional[int] = None,
    prompt_version_id: Optional[int] = None,
    pagination: PaginationParams = None
) -> tuple[List[Dict[str, Any]], int]:
    """Get paginated list of evaluations"""
    # Try file-based query first (new approach)
    if file_id is not None:
        query = db.table('evaluations').select('*, file:dataset_files(*), word_evaluations(*), prompt_version:prompt_versions(*)')
        query = query.eq('file_id', file_id)
    else:
        # Fallback to image-based query (backward compatibility)
        query = db.table('evaluations').select('*, image:images(*), word_evaluations(*), prompt_version:prompt_versions(*)')
        if image_id:
            query = query.eq('image_id', image_id)
    
    if prompt_version_id:
        query = query.eq('prompt_version_id', prompt_version_id)
    
    # Get total count
    count_result = query.execute()
    total = len(count_result.data) if count_result.data else 0
    
    # Apply pagination
    if pagination:
        query = query.range(pagination.skip, pagination.skip + pagination.limit - 1)
    
    result = query.execute()
    return result.data or [], total

async def update_evaluation(
    db: Client, 
    evaluation_id: int, 
    evaluation_update: EvaluationUpdate
) -> Optional[Dict[str, Any]]:
    """Update an evaluation with word evaluations"""
    update_data = evaluation_update.dict(exclude_unset=True)
    
    # Handle word evaluations separately
    word_evaluations_data = update_data.pop('word_evaluations', None)
    
    # Update main evaluation fields
    update_data['updated_at'] = datetime.utcnow().isoformat()
    
    # Handle word evaluations
    if word_evaluations_data is not None:
        # Delete existing word evaluations
        db.table('word_evaluations').delete().eq('evaluation_id', evaluation_id).execute()
        
        # Create new word evaluations
        word_eval_data_list = []
        for word_eval_data in word_evaluations_data:
            # Convert WordEvaluationCreate to dict if it's not already
            if hasattr(word_eval_data, 'dict'):
                word_eval_dict = word_eval_data.dict()
            else:
                word_eval_dict = word_eval_data
            
            word_eval_dict['evaluation_id'] = evaluation_id
            word_eval_data_list.append(word_eval_dict)
        
        if word_eval_data_list:
            db.table('word_evaluations').insert(word_eval_data_list).execute()
        
        # Also store as JSON for quick access
        update_data['word_evaluations_json'] = json.dumps([
            word_eval_data.dict() if hasattr(word_eval_data, 'dict') else word_eval_data 
            for word_eval_data in word_evaluations_data
        ])
    
    result = db.table('evaluations').update(update_data).eq('id', evaluation_id).execute()
    return result.data[0] if result.data else None

# Prompt Template CRUD operations
async def create_prompt_template(db: Client, template: PromptTemplateCreate) -> Dict[str, Any]:
    """Create a new prompt template"""
    # If this template is set as active, deactivate others
    if template.is_active:
        db.table('prompt_templates').update({'is_active': False}).eq('is_active', True).execute()
    
    template_data = template.dict()
    template_data['created_at'] = datetime.utcnow().isoformat()
    template_data['updated_at'] = datetime.utcnow().isoformat()
    
    result = db.table('prompt_templates').insert(template_data).execute()
    return result.data[0] if result.data else None

async def get_prompt_template(db: Client, template_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific prompt template"""
    result = db.table('prompt_templates').select('*').eq('id', template_id).execute()
    return result.data[0] if result.data else None

async def get_active_prompt_template(db: Client) -> Optional[Dict[str, Any]]:
    """Get the currently active prompt template"""
    result = db.table('prompt_templates').select('*').eq('is_active', True).execute()
    return result.data[0] if result.data else None

async def get_prompt_templates(db: Client) -> List[Dict[str, Any]]:
    """Get all prompt templates"""
    result = db.table('prompt_templates').select('*').order('created_at', desc=True).execute()
    return result.data or []

# Statistics and analytics
async def get_evaluation_stats(db: Client) -> Dict[str, Any]:
    """Get evaluation statistics"""
    # Total images
    total_images_result = db.table('images').select('id', count='exact').execute()
    total_images = total_images_result.count or 0
    
    # Total evaluations
    total_evaluations_result = db.table('evaluations').select('id', count='exact').execute()
    total_evaluations = total_evaluations_result.count or 0
    
    # Status counts
    pending_result = db.table('evaluations').select('id', count='exact').eq('processing_status', 'pending').execute()
    pending = pending_result.count or 0
    
    success_result = db.table('evaluations').select('id', count='exact').eq('processing_status', 'success').execute()
    successful = success_result.count or 0
    
    failed_result = db.table('evaluations').select('id', count='exact').eq('processing_status', 'failed').execute()
    failed = failed_result.count or 0
    
    # Average accuracy
    avg_accuracy_result = db.table('evaluations').select('accuracy').eq('processing_status', 'success').not_.is_('accuracy', 'null').execute()
    successful_accuracies = [eval['accuracy'] for eval in avg_accuracy_result.data if eval['accuracy'] is not None]
    avg_accuracy = sum(successful_accuracies) / len(successful_accuracies) if successful_accuracies else None
    
    # Accuracy by prompt version
    accuracy_by_version_result = db.table('evaluations').select('prompt_version, accuracy').eq('processing_status', 'success').not_.is_('accuracy', 'null').execute()
    
    # Group by prompt version and calculate average
    version_accuracies = {}
    for eval in accuracy_by_version_result.data:
        version = eval['prompt_version']
        accuracy = eval['accuracy']
        if version not in version_accuracies:
            version_accuracies[version] = []
        version_accuracies[version].append(accuracy)
    
    accuracy_by_version = {
        version: sum(accuracies) / len(accuracies) 
        for version, accuracies in version_accuracies.items()
    }
    
    return {
        "total_images": total_images,
        "total_evaluations": total_evaluations,
        "pending_evaluations": pending,
        "successful_evaluations": successful,
        "failed_evaluations": failed,
        "average_accuracy": float(avg_accuracy) if avg_accuracy else None,
        "accuracy_by_prompt_version": accuracy_by_version
    }

async def get_accuracy_distribution(db: Client) -> Dict[str, int]:
    """Get accuracy distribution"""
    # Get all successful evaluations with accuracy
    successful_evals_result = db.table('evaluations').select('accuracy').eq('processing_status', 'success').not_.is_('accuracy', 'null').execute()
    
    accuracies = [eval['accuracy'] for eval in successful_evals_result.data if eval['accuracy'] is not None]
    
    high = len([acc for acc in accuracies if acc >= 90])
    medium = len([acc for acc in accuracies if 70 <= acc < 90])
    low = len([acc for acc in accuracies if acc < 70])
    
    return {
        "high_accuracy": high,
        "medium_accuracy": medium,
        "low_accuracy": low,
        "total_processed": len(accuracies)
    }

# CSV Import functionality
async def import_csv_data(db: Client, csv_file_path: str, overwrite_existing: bool = False) -> Dict[str, Any]:
    """Import data from CSV file into the database"""
    imported_count = 0
    updated_count = 0
    errors = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header
                try:
                    # Extract required fields
                    number = row.get('#', '').strip()
                    url = row.get('Link', '').strip()
                    reference_text = row.get('Text', '').strip()
                    local_image = row.get('Local Image', '').strip()
                    
                    if not number or not reference_text:
                        errors.append(f"Row {row_num}: Missing required fields (# or Text)")
                        continue
                    
                    # Check if image already exists
                    existing_image = await get_image_by_number(db, number)
                    
                    if existing_image:
                        if overwrite_existing:
                            # Update existing image
                            update_data = ImageUpdate(
                                reference_text=reference_text,
                                url=url,
                                local_path=local_image
                            )
                            await update_image(db, existing_image['id'], update_data)
                            updated_count += 1
                        else:
                            # Skip existing
                            continue
                    else:
                        # Create new image
                        image_data = ImageCreate(
                            number=number,
                            url=url,
                            reference_text=reference_text,
                            local_path=local_image
                        )
                        await create_image(db, image_data)
                        imported_count += 1
                    
                except Exception as e:
                    errors.append(f"Row {row_num}: {str(e)}")
                    continue
        
        return {
            "imported_count": imported_count,
            "updated_count": updated_count,
            "errors": errors,
            "message": f"Import completed. {imported_count} new images, {updated_count} updated."
        }
    
    except FileNotFoundError:
        return {
            "imported_count": 0,
            "updated_count": 0,
            "errors": [f"CSV file not found: {csv_file_path}"],
            "message": "Import failed - file not found"
        }
    except Exception as e:
        return {
            "imported_count": 0,
            "updated_count": 0,
            "errors": [f"Import error: {str(e)}"],
            "message": "Import failed"
        }

async def import_csv_data_into_dataset(db: Client, csv_file_path: str, dataset_id: int, overwrite_existing: bool = False) -> Dict[str, Any]:
    """Import data from CSV file into the database"""
    logging.info(f"[CRUD] Starting CSV import for dataset_id={dataset_id}, file={csv_file_path}")
    
    # Debug: Check if dataset_images table exists
    table_debug = await debug_dataset_images_table(db)
    logging.info(f"[CRUD] dataset_images table debug: {table_debug}")
    
    imported_count = 0
    updated_count = 0
    errors = []
    dataset = await get_dataset(db, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")
    
    logging.info(f"[CRUD] Found dataset: {dataset.get('name', 'Unknown')}")
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header
                try:
                    logging.info(f"[CRUD] Processing row {row_num}: {row}")
                    
                    # Extract required fields
                    number = row.get('#', '').strip()
                    url = row.get('URL', row.get('Link', '')).strip()  # Try URL first, then Link as fallback
                    reference_text = row.get('Text', '').strip()
                    human_evaluation_text = row.get('Human Evaluation Text', '').strip()
                    local_image = row.get('Local Image', '').strip()
                    
                    if not number or not reference_text:
                        errors.append(f"Row {row_num}: Missing required fields (# or Text)")
                        logging.warning(f"[CRUD] Row {row_num}: Missing required fields")
                        continue
                    
                    logging.info(f"[CRUD] Creating image with number={number}, reference_text={reference_text[:50]}...")
                    
                    # Create new image
                    image_data = ImageCreate(
                        number=number,
                        url=url,
                        reference_text=reference_text,
                        human_evaluation_text=human_evaluation_text,
                        local_path=local_image
                    )
                    db_image = await create_image(db, image_data)
                    
                    if not db_image:
                        errors.append(f"Row {row_num}: Failed to create image")
                        logging.error(f"[CRUD] Row {row_num}: Failed to create image")
                        continue
                    
                    logging.info(f"[CRUD] Created image with id={db_image['id']}")
                    
                    # Associate image with dataset
                    association_success = await associate_image_with_dataset(db, db_image['id'], dataset_id)
                    
                    if association_success:
                        imported_count += 1
                        logging.info(f"[CRUD] Successfully associated image_id={db_image['id']} with dataset_id={dataset_id}")
                    else:
                        errors.append(f"Row {row_num}: Failed to associate image with dataset")
                        logging.error(f"[CRUD] Row {row_num}: Failed to associate image_id={db_image['id']} with dataset_id={dataset_id}")
                        
                except Exception as e:
                    error_msg = f"Row {row_num}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(f"[CRUD] {error_msg}")
                    continue
        
        logging.info(f"[CRUD] Import completed. imported_count={imported_count}, updated_count={updated_count}, errors={len(errors)}")
        
        # Update dataset with correct image count
        await update_dataset(db, dataset_id, DatasetUpdate(
            image_count=dataset['image_count'] + imported_count,
            status=DatasetStatus.VALIDATED
        ))

        return {
            "imported_count": imported_count,
            "updated_count": updated_count,
            "errors": errors,
            "message": f"Import completed. {imported_count} new images, {updated_count} updated."
        }
    
    except FileNotFoundError:
        logging.error(f"[CRUD] CSV file not found: {csv_file_path}")
        return {
            "imported_count": 0,
            "updated_count": 0,
            "errors": [f"CSV file not found: {csv_file_path}"],
            "message": "Import failed - file not found"
        }
    except Exception as e:
        logging.error(f"[CRUD] Import error: {str(e)}")
        return {
            "imported_count": 0,
            "updated_count": 0,
            "errors": [f"Import error: {str(e)}"],
            "message": "Import failed"
        }

async def get_latest_imported_images(db: Client, count: int) -> List[int]:
    """Get IDs of the most recently imported images"""
    result = db.table('images').select('id').order('created_at', desc=True).limit(count).execute()
    return [row['id'] for row in result.data] if result.data else []

async def debug_dataset_images_table(db: Client) -> Dict[str, Any]:
    """Debug function to check dataset_images table structure"""
    try:
        # Try to select from the table to see if it exists
        result = db.table('dataset_images').select('*').limit(1).execute()
        logging.info(f"[CRUD] dataset_images table exists, sample data: {result.data}")
        return {"exists": True, "sample_data": result.data}
    except Exception as e:
        logging.error(f"[CRUD] Error accessing dataset_images table: {str(e)}")
        return {"exists": False, "error": str(e)}

async def associate_image_with_dataset(db: Client, image_id: int, dataset_id: int) -> bool:
    """Associate an image with a dataset"""
    logging.info(f"[CRUD] Associating image_id={image_id} with dataset_id={dataset_id}")
    
    try:
        # Check if association already exists
        existing_result = db.table('dataset_images').select('*').eq('image_id', image_id).eq('dataset_id', dataset_id).execute()
        if existing_result.data:
            logging.info(f"[CRUD] Association already exists for image_id={image_id}, dataset_id={dataset_id}")
            return True  # Already associated
        
        # Create association
        association_data = {
            'image_id': image_id,
            'dataset_id': dataset_id
        }
        
        logging.info(f"[CRUD] Creating association with data: {association_data}")
        result = db.table('dataset_images').insert(association_data).execute()
        
        success = len(result.data) > 0 if result.data else False
        logging.info(f"[CRUD] Association creation result: {success}, data: {result.data}")
        
        if not success:
            logging.error(f"[CRUD] Failed to create association for image_id={image_id}, dataset_id={dataset_id}")
            if hasattr(result, 'error'):
                logging.error(f"[CRUD] Database error: {result.error}")
        
        return success
        
    except Exception as e:
        logging.error(f"[CRUD] Exception in associate_image_with_dataset: {str(e)}")
        return False

async def is_image_in_dataset(db: Client, image_id: int, dataset_id: int) -> bool:
    """Check if an image is already associated with a dataset"""
    result = db.table('dataset_images').select('*').eq('image_id', image_id).eq('dataset_id', dataset_id).execute()
    return len(result.data) > 0 if result.data else False

async def get_dataset_images(db: Client, dataset_id: int) -> List[Dict[str, Any]]:
    """Get all images associated with a dataset"""
    # Query through the junction table to get image IDs
    junction_result = db.table('dataset_images').select('image_id').eq('dataset_id', dataset_id).execute()
    image_ids = [row['image_id'] for row in junction_result.data] if junction_result.data else []
    
    if not image_ids:
        return []
    
    # Get the actual image data
    result = db.table('images').select('*').in_('id', image_ids).order('id').execute()
    return result.data or []

# New Dataset CRUD operations
async def create_dataset(db: Client, dataset: DatasetCreate) -> Dict[str, Any]:
    """Create a new dataset"""
    dataset_data = dataset.dict()
    dataset_data['created_at'] = datetime.utcnow().isoformat()
    dataset_data['updated_at'] = datetime.utcnow().isoformat()
    dataset_data['file_count'] = 0  # Changed from image_count to file_count
    # input_schema is included in dataset_data if present
    result = db.table('datasets').insert(dataset_data).execute()
    return result.data[0] if result.data else None

async def get_dataset(db: Client, dataset_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific dataset with its files"""
    # First get the dataset
    result = db.table('datasets').select('*').eq('id', dataset_id).execute()
    dataset = result.data[0] if result.data else None
    
    if not dataset:
        return None
    
    # Get associated files through the junction table
    files = await get_dataset_files_by_dataset(db, dataset_id)
    
    # Return files in the correct format for DatasetWithFiles model
    # The DatasetFile model expects expected_output, not reference_text
    dataset['files'] = files
    
    # For backward compatibility, also include images (empty for now)
    dataset['images'] = []
    
    # input_schema is already included if present in the DB
    return dataset

async def get_datasets(db: Client, user_id: str) -> List[Dict[str, Any]]:
    """Get all datasets for a user"""
    result = db.table('datasets').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
    return result.data or []

async def update_dataset(db: Client, dataset_id: int, dataset_update: DatasetUpdate) -> Optional[Dict[str, Any]]:
    """Update a dataset"""
    update_data = dataset_update.dict(exclude_unset=True)
    update_data['updated_at'] = datetime.utcnow().isoformat()
    # input_schema is included in update_data if present
    result = db.table('datasets').update(update_data).eq('id', dataset_id).execute()
    return result.data[0] if result.data else None

async def process_dataset_upload(db: Client, dataset_id: int, images_zip, reference_csv) -> Dict[str, Any]:
    """Process uploaded ZIP of images and CSV with reference texts"""
    dataset = await get_dataset(db, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract ZIP file
        zip_path = os.path.join(temp_dir, "images.zip")
        with open(zip_path, "wb") as f:
            f.write(await images_zip.read())
        
        # Save CSV file
        csv_path = os.path.join(temp_dir, "reference.csv")
        with open(csv_path, "wb") as f:
            f.write(await reference_csv.read())
        
        # Extract images
        images_dir = os.path.join(temp_dir, "images")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        
        # Read CSV and validate
        image_refs = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                filename = row.get('image_filename', '').strip()
                reference_text = row.get('reference_text', '').strip()
                if filename and reference_text:
                    image_refs[filename] = reference_text
        
        # Get list of extracted image files
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(file)
        
        # Validate that all images have reference text and vice versa
        missing_refs = [img for img in image_files if img not in image_refs]
        missing_images = [ref for ref in image_refs.keys() if ref not in image_files]
        
        if missing_refs or missing_images:
            raise ValueError(f"Validation failed. Missing references: {missing_refs}, Missing images: {missing_images}")
        
        # Create Image records and associate with dataset
        created_images = []
        for filename, reference_text in image_refs.items():
            # Create a unique number for the image
            image_number = f"{dataset['name']}_{filename}"
            
            # Check if image already exists
            existing = await get_image_by_number(db, image_number)
            if not existing:
                image_data = ImageCreate(
                    number=image_number,
                    url=f"/datasets/{dataset_id}/images/{filename}",
                    reference_text=reference_text,
                    local_path=os.path.join(images_dir, filename)
                )
                db_image = await create_image(db, image_data)
                created_images.append(db_image)
        
        # Associate images with dataset
        for image in created_images:
            await associate_image_with_dataset(db, image['id'], dataset_id)
        
        # Update dataset metadata
        await update_dataset(db, dataset_id, DatasetUpdate(
            image_count=len(created_images),
            status=DatasetStatus.VALIDATED
        ))
        
        # Get updated dataset
        updated_dataset = await get_dataset(db, dataset_id)
        return updated_dataset

# Prompt Family CRUD operations
async def create_prompt_family(db: Client, family: PromptFamilyCreate) -> Dict[str, Any]:
    """Create a new prompt family"""
    family_data = family.dict()
    family_data['created_at'] = datetime.utcnow().isoformat()
    
    result = db.table('prompt_families').insert(family_data).execute()
    return result.data[0] if result.data else None

async def get_prompt_family(db: Client, family_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific prompt family with its versions"""
    result = db.table('prompt_families').select('*, prompt_versions(*)').eq('id', family_id).execute()
    return result.data[0] if result.data else None

async def get_prompt_families(db: Client, user_id: str) -> List[Dict[str, Any]]:
    """Get all prompt families for a user"""
    result = db.table('prompt_families').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
    return result.data or []

async def update_prompt_family(db: Client, family_id: int, family_update: PromptFamilyCreate) -> Optional[Dict[str, Any]]:
    """Update an existing prompt family"""
    update_data = family_update.dict(exclude_unset=True)
    update_data['updated_at'] = datetime.utcnow().isoformat()
    
    result = db.table('prompt_families').update(update_data).eq('id', family_id).execute()
    return result.data[0] if result.data else None

# Prompt Version CRUD operations
async def get_prompt_versions(db: Client, family_id: int, user_id: str) -> List[Dict[str, Any]]:
    """Get all versions for a prompt family"""
    result = db.table('prompt_versions').select('*').eq('family_id', family_id).eq('user_id', user_id).order('created_at', desc=True).execute()
    return result.data or []

async def get_prompt_version(db: Client, version_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific prompt version by ID"""
    result = db.table('prompt_versions').select('*').eq('id', version_id).execute()
    return result.data[0] if result.data else None

async def generate_next_version(db: Client, family_id: int, version_type: VersionType) -> str:
    """Generate the next semantic version number"""
    # Get latest version for this family
    result = db.table('prompt_versions').select('version').eq('family_id', family_id).order('created_at', desc=True).limit(1).execute()
    latest_version = result.data[0]['version'] if result.data else None
    
    if not latest_version:
        return "1.0.0"
    
    # Parse version (assume format: major.minor.patch)
    try:
        major, minor, patch = map(int, latest_version.split('.'))
    except ValueError:
        return "1.0.0"
    
    # Increment based on type
    if version_type == VersionType.MAJOR:
        major += 1
        minor = 0
        patch = 0
    elif version_type == VersionType.MINOR:
        minor += 1
        patch = 0
    elif version_type == VersionType.PATCH:
        patch += 1
    
    return f"{major}.{minor}.{patch}"

async def create_prompt_version(db: Client, version_data: PromptVersionCreate) -> Dict[str, Any]:
    """Create a new prompt version"""
    # Convert Pydantic model to dict and exclude version_type
    version_dict = version_data.dict(exclude={'version_type'})
    
    # Ensure issues is always a JSON string
    issues = version_dict.get("issues", [])
    if not isinstance(issues, str):
        version_dict["issues"] = json.dumps(issues)
    
    version_dict['created_at'] = datetime.utcnow().isoformat()
    
    result = db.table('prompt_versions').insert(version_dict).execute()
    return result.data[0] if result.data else None

async def update_prompt_version(db: Client, version_id: int, version_update: PromptVersionUpdate) -> Optional[Dict[str, Any]]:
    """Update an existing prompt version"""
    update_data = version_update.dict(exclude_unset=True)
    # Ensure issues is always a JSON string
    if "issues" in update_data and not isinstance(update_data["issues"], str):
        update_data["issues"] = json.dumps(update_data["issues"])
    result = db.table('prompt_versions').update(update_data).eq('id', version_id).execute()
    return result.data[0] if result.data else None

async def promote_prompt_version(db: Client, version_id: int) -> bool:
    """Promote a prompt version to production"""
    # Get the version to promote
    version_result = db.table('prompt_versions').select('*').eq('id', version_id).execute()
    if not version_result.data:
        return False
    
    db_version = version_result.data[0]
    
    # Set all other versions in this family to archived
    db.table('prompt_versions').update({'status': 'archived'}).eq('family_id', db_version['family_id']).eq('status', 'production').execute()
    
    # Promote this version
    db.table('prompt_versions').update({'status': 'production'}).eq('id', version_id).execute()
    
    # Update family's production version reference
    db.table('prompt_families').update({'production_version': db_version['version']}).eq('id', db_version['family_id']).execute()
    
    return True

# Evaluation Run CRUD operations
async def create_evaluation_run(db: Client, run: EvaluationRunCreate) -> Dict[str, Any]:
    """Create a new evaluation run"""
    logging.info("[CRUD] Entered create_evaluation_run")
    try:
        run_data = {
            'name': run.name,
            'description': run.description,
            'hypothesis': run.hypothesis,
            'user_id': run.user_id,
            'status': 'pending',
            'progress_percentage': 0,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        result = db.table('evaluation_runs').insert(run_data).execute()
        db_run = result.data[0] if result.data else None
        
        if not db_run:
            raise Exception("Failed to create evaluation run")
        
        logging.info(f"[CRUD] Created EvaluationRun object with id: {db_run['id']}")
        
        # Add datasets to evaluation run
        for dataset_id in run.dataset_ids:
            logging.info(f"[CRUD] Adding dataset_id: {dataset_id} to run {db_run['id']}")
            dataset = await get_dataset(db, dataset_id)
            logging.info("[Crud]!!Dataset Found")
            if dataset:
                db.table('evaluation_run_datasets').insert({
                    'evaluation_run_id': db_run['id'],
                    'dataset_id': dataset_id
                }).execute()
            else:
                logging.warning(f"[CRUD] Dataset {dataset_id} not found when adding to run {db_run['id']}")
        
        logging.info(f"[CRUD] Datasets Added to run {db_run['id']}")
        
        # Add prompt configurations
        for config in run.prompt_configurations:
            logging.info(f"[CRUD] Adding prompt config: family_id={config.family_id}, version_id={config.version_id}, label={config.label}")
            # Find the prompt version by id
            version_result = db.table('prompt_versions').select('*').eq('id', config.version_id).execute()
            version = version_result.data[0] if version_result.data else None
            
            if version:
                run_prompt_data = {
                    'evaluation_run_id': db_run['id'],
                    'prompt_version_id': version['id'],
                    'label': config.label
                }
                db.table('evaluation_run_prompts').insert(run_prompt_data).execute()
                logging.info(f"[CRUD] Added EvaluationRunPrompt for version_id={version['id']} to run {db_run['id']}")
            else:
                logging.warning(f"[CRUD] PromptVersion not found for id={config.version_id}")
        
        logging.info(f"[CRUD] Committed and refreshed EvaluationRun with id: {db_run['id']}")
        
        # Get the complete run with relationships
        complete_run = await get_evaluation_run(db, db_run['id'])
        return complete_run or db_run
        
    except Exception as e:
        logging.exception(f"[CRUD] Exception in create_evaluation_run: {str(e)}")
        raise

async def get_evaluation_runs(db: Client, user_id: str) -> List[Dict[str, Any]]:
    """Get all evaluation runs for a user"""
    result = db.table('evaluation_runs').select('''
        *,
        evaluation_run_prompts(
            *,
            prompt_version:prompt_versions(
                id,
                family_id,
                version,
                prompt_text
            )
        )
    ''').eq('user_id', user_id).order('created_at', desc=True).execute()
    runs = result.data or []
    
    # For each run, get its datasets through the junction table
    for run in runs:
        run_id = run['id']
        dataset_junction_result = db.table('evaluation_run_datasets').select('dataset_id').eq('evaluation_run_id', run_id).execute()
        dataset_ids = [row['dataset_id'] for row in dataset_junction_result.data] if dataset_junction_result.data else []
        
        # Get basic dataset info (without images to avoid the relationship issue)
        datasets = []
        for dataset_id in dataset_ids:
            dataset_result = db.table('datasets').select('*').eq('id', dataset_id).execute()
            if dataset_result.data:
                datasets.append(dataset_result.data[0])
        
        run['datasets'] = datasets
        run['dataset_ids'] = [dataset['id'] for dataset in datasets]
        
        # Rename evaluation_run_prompts to prompt_configurations for frontend compatibility
        run['prompt_configurations'] = run.pop('evaluation_run_prompts', [])
        for prompt_config in run.get('prompt_configurations', []):
            if prompt_config.get('prompt_version'):
                prompt_config['family_id'] = prompt_config['prompt_version']['family_id']
                prompt_config['version'] = prompt_config['prompt_version']['version']
                prompt_config['prompt_text'] = prompt_config['prompt_version']['prompt_text']
                prompt_config['prompt_version_id'] = prompt_config['prompt_version']['id']
                prompt_config['version_id'] = prompt_config['prompt_version']['id']
            else:
                prompt_config['family_id'] = None
                prompt_config['version'] = None
                prompt_config['prompt_text'] = None
                prompt_config['prompt_version_id'] = None
                prompt_config['version_id'] = None
    
    return runs

async def get_evaluation_run(db: Client, run_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific evaluation run with full details"""
    # First get the evaluation run with its evaluations and their relationships
    result = db.table('evaluation_runs').select('''
        *,
        evaluation_run_prompts!inner(
            *,
            prompt_version:prompt_versions!inner(*)
        ),
        evaluations(
            *,
            word_evaluations(*),
            image:images(*),
            file:dataset_files(*)
        )
    ''').eq('id', run_id).execute()
    run = result.data[0] if result.data else None
    
    if not run:
        return None
    
    logging.info(f"[EVAL-RUN {run_id}] Raw run data: {json.dumps(run, indent=2)}")
    
    # Get datasets for this run through the junction table
    dataset_junction_result = db.table('evaluation_run_datasets').select('dataset_id').eq('evaluation_run_id', run_id).execute()
    dataset_ids = [row['dataset_id'] for row in dataset_junction_result.data] if dataset_junction_result.data else []
    
    # Get the actual datasets with their images
    datasets = []
    for dataset_id in dataset_ids:
        dataset = await get_dataset(db, dataset_id)
        if dataset:
            datasets.append(dataset)
    
    run['datasets'] = datasets
    
    # Add dataset_ids to the run for Pydantic schema compatibility
    run['dataset_ids'] = [dataset['id'] for dataset in datasets]
    
    # Format prompt configurations to match the schema
    # Rename evaluation_run_prompts to prompt_configurations for frontend compatibility
    run['prompt_configurations'] = run.pop('evaluation_run_prompts', [])
    for prompt_config in run.get('prompt_configurations', []):
        logging.info(f"[EVAL-RUN {run_id}] Processing prompt config: {json.dumps(prompt_config, indent=2)}")
        if prompt_config.get('prompt_version'):
            version = prompt_config['prompt_version']
            logging.info(f"[EVAL-RUN {run_id}] Found version data: {json.dumps(version, indent=2)}")
            prompt_config.update({
                'family_id': version.get('family_id'),
                'version': version.get('version'),
                'prompt_text': version.get('prompt_text'),
                'prompt_version_id': version.get('id'),
                'version_id': version.get('id')
            })
        else:
            logging.warning(f"[EVAL-RUN {run_id}] No prompt_version found for config: {json.dumps(prompt_config, indent=2)}")
            prompt_config.update({
                'family_id': None,
                'version': None,
                'prompt_text': None,
                'prompt_version_id': None,
                'version_id': None
            })
    
    logging.info(f"[EVAL-RUN {run_id}] Final prompt configurations: {json.dumps(run['prompt_configurations'], indent=2)}")
    return run

async def get_evaluation_comparison(db: Client, run_id: int) -> Optional[Dict[str, Any]]:
    """Generate comparison results for an evaluation run"""
    run = await get_evaluation_run(db, run_id)
    if not run or run.get('status') != 'success':
        return None
    
    # Group evaluations by prompt version
    evaluations_by_version = {}
    for evaluation in run.get('evaluations', []):
        # Get the prompt version string from prompt_version_id
        prompt_version_id = evaluation.get('prompt_version_id')
        prompt_version = None
        
        if prompt_version_id:
            # Fetch the prompt version string from the prompt_versions table
            prompt_version_result = db.table('prompt_versions').select('version').eq('id', prompt_version_id).single().execute()
            if prompt_version_result.data:
                prompt_version = prompt_version_result.data.get('version')
        
        if prompt_version:
            if prompt_version not in evaluations_by_version:
                evaluations_by_version[prompt_version] = []
            evaluations_by_version[prompt_version].append(evaluation)
    
    # Calculate summary metrics for each prompt version
    summary_metrics = []
    for prompt_version, evaluations in evaluations_by_version.items():
        if not evaluations:
            continue
            
        # Find the prompt configuration for this version
        prompt_config = None
        for config in run.get('prompt_configurations', []):
            if config.get('version') == prompt_version:
                prompt_config = config
                break
        
        # Calculate metrics
        total_accuracy = sum(eval.get('accuracy', 0) or 0 for eval in evaluations)
        avg_accuracy = total_accuracy / len(evaluations) if evaluations else 0
        
        total_correct_words = sum(eval.get('correct_words', 0) or 0 for eval in evaluations)
        total_words = sum(eval.get('total_words', 0) or 0 for eval in evaluations)
        character_error_rate = 1 - (total_correct_words / total_words) if total_words > 0 else 0
        
        # Calculate average latency (placeholder - would need to be stored)
        avg_latency_ms = 0
        
        # Estimate cost (placeholder - would need actual cost tracking)
        estimated_cost_per_1k = 0.01  # Placeholder
        
        # Error breakdown (placeholder - would need detailed error analysis)
        error_breakdown = {
            "character_errors": 0,
            "word_boundary_errors": 0,
            "spacing_errors": 0
        }
        
        summary_metrics.append({
            "prompt_version": prompt_version,
            "label": prompt_config.get('label') if prompt_config else f"Version {prompt_version}",
            "overall_accuracy": round(avg_accuracy, 2),
            "character_error_rate": round(character_error_rate, 4),
            "avg_latency_ms": avg_latency_ms,
            "estimated_cost_per_1k": estimated_cost_per_1k,
            "error_breakdown": error_breakdown
        })
    
    # Generate word-level comparisons
    word_comparisons = []
    
    # Group evaluations by image_id to compare same images across versions
    evaluations_by_image = {}
    for evaluation in run.get('evaluations', []):
        image_id = evaluation.get('image_id')
        if image_id not in evaluations_by_image:
            evaluations_by_image[image_id] = {}
        
        # Get the prompt version string from prompt_version_id
        prompt_version_id = evaluation.get('prompt_version_id')
        prompt_version = None
        
        if prompt_version_id:
            # Fetch the prompt version string from the prompt_versions table
            prompt_version_result = db.table('prompt_versions').select('version').eq('id', prompt_version_id).single().execute()
            if prompt_version_result.data:
                prompt_version = prompt_version_result.data.get('version')
        
        if prompt_version:
            evaluations_by_image[image_id][prompt_version] = evaluation
    
    # Compare evaluations for the same image across different versions
    for image_id, version_evaluations in evaluations_by_image.items():
        if len(version_evaluations) < 2:
            continue  # Need at least 2 versions to compare
            
        # Get the image info
        image = None
        for dataset in run.get('datasets', []):
            for img in dataset.get('images', []):
                if img.get('id') == image_id:
                    image = img
                    break
            if image:
                break
        
        # Compare word-level results (simplified - would need actual word-level data)
        # For now, create a basic comparison based on overall accuracy
        versions = list(version_evaluations.keys())
        if len(versions) >= 2:
            eval1 = version_evaluations[versions[0]]
            eval2 = version_evaluations[versions[1]]
            
            # Determine which performed better
            if eval1.get('accuracy', 0) > eval2.get('accuracy', 0):
                winner = versions[0]
                status = "improved"
            elif eval2.get('accuracy', 0) > eval1.get('accuracy', 0):
                winner = versions[1]
                status = "improved"
            else:
                winner = "tie"
                status = "match"
            
            word_comparisons.append({
                "image_filename": f"image_{image.get('number')}" if image else f"image_{image_id}",
                "word_index": 0,  # Placeholder
                "reference_word": "overall_performance",
                "control_output": f"{eval1.get('accuracy', 0)}% accuracy",
                "variation_output": f"{eval2.get('accuracy', 0)}% accuracy",
                "status": status,
                "error_type": None
            })
    
    # Determine overall winner
    if summary_metrics:
        best_metric = max(summary_metrics, key=lambda x: x["overall_accuracy"])
        winner = best_metric["label"]
        confidence_level = 0.95 if len(summary_metrics) > 1 else 0.5
    else:
        winner = None
        confidence_level = None
    
    return {
        "evaluation_run_id": run_id,
        "summary_metrics": summary_metrics,
        "word_comparisons": word_comparisons,
        "winner": winner,
        "confidence_level": confidence_level
    }

# API Log CRUD operations
async def create_api_log(db: Client, log: APILogCreate) -> Dict[str, Any]:
    """Create a new API log entry"""
    log_data = log.dict()
    log_data['created_at'] = datetime.utcnow().isoformat()
    
    result = db.table('api_logs').insert(log_data).execute()
    return result.data[0] if result.data else None

async def get_api_logs_for_user(db: Client, user_id: str, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Get API logs for a user"""
    result = db.table('api_logs').select('*').eq('user_id', user_id).order('created_at', desc=True).range(skip, skip + limit - 1).execute()
    return result.data or []

# API Key CRUD operations
async def create_api_key(db: Client, key_data: APIKeyCreate) -> Dict[str, Any]:
    """Create a new API key"""
    import secrets
    import hashlib
    
    # Generate a secure API key
    key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    key_record = {
        'key_name': key_data.key_name,
        'key_hash': key_hash,
        'key_preview': key[-4:],  # Last 4 characters for display
        'is_active': True,
        'created_at': datetime.utcnow().isoformat()
    }
    
    result = db.table('api_keys').insert(key_record).execute()
    db_key = result.data[0] if result.data else None
    
    if db_key:
        # Return the key with the actual key value (only time we show it)
        db_key['actual_key'] = key
    
    return db_key

async def get_api_keys(db: Client) -> List[Dict[str, Any]]:
    """Get all active API keys"""
    result = db.table('api_keys').select('*').eq('is_active', True).order('created_at', desc=True).execute()
    return result.data or []

async def revoke_api_key(db: Client, key_id: int) -> bool:
    """Revoke an API key"""
    result = db.table('api_keys').update({'is_active': False}).eq('id', key_id).execute()
    return len(result.data) > 0 if result.data else False

async def get_api_key_usage(db: Client, key_id: int) -> Optional[Dict[str, Any]]:
    """Get usage statistics for an API key"""
    # This would be implemented with actual usage tracking
    # For now, return placeholder data
    return {
        "api_key_id": key_id,
        "total_calls": 0,
        "calls_today": 0,
        "calls_this_month": 0,
        "error_rate": 0.0,
        "avg_response_time_ms": 0
    }

# Historical Analysis functions
async def get_performance_trends(
    db: Client, 
    prompt_family_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    days_back: int = 30
) -> List[Dict[str, Any]]:
    """Get performance trends over time"""
    since_date = datetime.utcnow() - timedelta(days=days_back)
    
    # This would be implemented to analyze historical performance
    # For now, return placeholder data
    return []

async def get_regression_alerts(db: Client) -> List[Dict[str, Any]]:
    """Get active regression alerts"""
    # This would check for performance regressions
    # For now, return empty list
    return []

async def get_evaluation_run_progress(db: Client, run_id: int) -> Optional[Dict[str, Any]]:
    """Get real-time progress for an evaluation run"""
    result = db.table('evaluation_runs').select('*').eq('id', run_id).execute()
    run = result.data[0] if result.data else None
    
    if not run:
        return None
    
    return {
        "evaluation_run_id": run_id,
        "overall_progress": run.get('progress_percentage', 0),
        "prompt_progress": {},  # Would track progress per prompt
        "current_image": run.get('current_step', ''),
        "log_entries": []
    }

async def delete_image_from_dataset(db: Client, dataset_id: int, image_id: int) -> bool:
    """Remove an image from a dataset, delete the association, and delete the image itself."""
    # Remove association from dataset_images table
    db.table('dataset_images').delete().eq('dataset_id', dataset_id).eq('image_id', image_id).execute()
    
    # Update dataset image count
    dataset_result = db.table('datasets').select('*').eq('id', dataset_id).execute()
    dataset = dataset_result.data[0] if dataset_result.data else None
    if dataset:
        new_count = max(0, dataset.get('image_count', 0) - 1)
        db.table('datasets').update({
            'image_count': new_count,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', dataset_id).execute()
    
    # Delete the image itself
    result = db.table('images').delete().eq('id', image_id).execute()
    return len(result.data) > 0 if result.data else False

async def get_prompt_version_by_version_string(db: Client, version: str, family_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific prompt version by its version string and family_id"""
    result = db.table('prompt_versions').select('*').eq('version', version).eq('family_id', family_id).execute()
    return result.data[0] if result.data else None

# Dataset File CRUD operations (new generic approach)
async def create_dataset_file(db: Client, file_data: DatasetFileCreate) -> Dict[str, Any]:
    """Create a new dataset file using Supabase"""
    file_dict = file_data.dict()
    file_dict['created_at'] = datetime.utcnow().isoformat()
    file_dict['updated_at'] = datetime.utcnow().isoformat()
    
    # Ensure user_id is included
    if 'user_id' not in file_dict or not file_dict['user_id']:
        raise ValueError("user_id is required for creating dataset files")
    
    result = db.table('dataset_files').insert(file_dict).execute()
    return result.data[0] if result.data else None

async def get_dataset_file(db: Client, file_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific dataset file with its evaluations"""
    result = db.table('dataset_files').select('*, evaluations(*)').eq('id', file_id).execute()
    return result.data[0] if result.data else None

async def get_dataset_file_by_number(db: Client, number: str) -> Optional[Dict[str, Any]]:
    """Get dataset file by number"""
    result = db.table('dataset_files').select('*').eq('number', number).execute()
    return result.data[0] if result.data else None

async def get_dataset_files(
    db: Client, 
    dataset_id: Optional[int] = None,
    file_type: Optional[str] = None,
    pagination: PaginationParams = None
) -> tuple[List[Dict[str, Any]], int]:
    """Get paginated list of dataset files with optional filters"""
    if dataset_id:
        # Get files through association table
        junction_result = db.table('dataset_file_associations').select('file_id').eq('dataset_id', dataset_id).execute()
        file_ids = [row['file_id'] for row in junction_result.data] if junction_result.data else []
        
        if not file_ids:
            return [], 0
        
        query = db.table('dataset_files').select('*, evaluations(*)').in_('id', file_ids)
    else:
        query = db.table('dataset_files').select('*, evaluations(*)')
    
    # Apply filters
    if file_type:
        query = query.eq('file_type', file_type)
    
    # Get total count
    count_result = query.execute()
    total = len(count_result.data) if count_result.data else 0
    
    # Apply pagination
    if pagination:
        query = query.range(pagination.skip, pagination.skip + pagination.limit - 1)
    
    result = query.execute()
    return result.data or [], total

async def update_dataset_file(db: Client, file_id: int, file_update: DatasetFileUpdate) -> Optional[Dict[str, Any]]:
    """Update an existing dataset file"""
    update_data = file_update.dict(exclude_unset=True)
    update_data['updated_at'] = datetime.utcnow().isoformat()
    
    # Handle field mapping for backward compatibility
    if 'reference_text' in update_data:
        update_data['expected_output'] = update_data.pop('reference_text')
    
    result = db.table('dataset_files').update(update_data).eq('id', file_id).execute()
    return result.data[0] if result.data else None

async def delete_dataset_file(db: Client, file_id: int) -> bool:
    """Delete a dataset file"""
    result = db.table('dataset_files').delete().eq('id', file_id).execute()
    return len(result.data) > 0 if result.data else False

async def associate_file_with_dataset(db: Client, file_id: int, dataset_id: int) -> bool:
    """Associate a dataset file with a dataset"""
    logging.info(f"[CRUD] Associating file_id={file_id} with dataset_id={dataset_id}")
    
    try:
        # Check if association already exists
        existing_result = db.table('dataset_file_associations').select('*').eq('file_id', file_id).eq('dataset_id', dataset_id).execute()
        if existing_result.data:
            logging.info(f"[CRUD] Association already exists for file_id={file_id}, dataset_id={dataset_id}")
            return True  # Already associated
        
        # Create association
        association_data = {
            'file_id': file_id,
            'dataset_id': dataset_id
        }
        
        logging.info(f"[CRUD] Creating association with data: {association_data}")
        result = db.table('dataset_file_associations').insert(association_data).execute()
        
        success = len(result.data) > 0 if result.data else False
        logging.info(f"[CRUD] Association creation result: {success}, data: {result.data}")
        
        return success
        
    except Exception as e:
        logging.error(f"[CRUD] Exception in associate_file_with_dataset: {str(e)}")
        return False

async def get_dataset_files_by_dataset(db: Client, dataset_id: int) -> List[Dict[str, Any]]:
    """Get all files associated with a dataset"""
    # Query through the junction table to get file IDs
    junction_result = db.table('dataset_file_associations').select('file_id').eq('dataset_id', dataset_id).execute()
    file_ids = [row['file_id'] for row in junction_result.data] if junction_result.data else []
    
    if not file_ids:
        return []
    
    # Get the actual file data
    result = db.table('dataset_files').select('*').in_('id', file_ids).order('id').execute()
    return result.data or []

async def is_file_in_dataset(db: Client, file_id: int, dataset_id: int) -> bool:
    """Check if a file is already associated with a dataset"""
    result = db.table('dataset_file_associations').select('*').eq('file_id', file_id).eq('dataset_id', dataset_id).execute()
    return len(result.data) > 0 if result.data else False

async def delete_file_from_dataset(db: Client, dataset_id: int, file_id: int) -> bool:
    """Remove a file from a dataset, delete the association, and delete the file itself."""
    # Remove association from dataset_file_associations table
    db.table('dataset_file_associations').delete().eq('dataset_id', dataset_id).eq('file_id', file_id).execute()
    
    # Update dataset file count
    dataset_result = db.table('datasets').select('*').eq('id', dataset_id).execute()
    dataset = dataset_result.data[0] if dataset_result.data else None
    if dataset:
        new_count = max(0, dataset.get('file_count', 0) - 1)
        db.table('datasets').update({
            'file_count': new_count,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', dataset_id).execute()
    
    # Delete the file itself
    result = db.table('dataset_files').delete().eq('id', file_id).execute()
    return len(result.data) > 0 if result.data else False

# New CSV import function for dataset files
async def import_csv_data_into_dataset_files(db: Client, csv_file_path: str, dataset_id: int, overwrite_existing: bool = False) -> Dict[str, Any]:
    """Import data from CSV file into dataset_files table - handles both image and PDF formats"""
    logging.info(f"[CRUD] Starting CSV import for dataset_id={dataset_id}, file={csv_file_path}")
    
    imported_count = 0
    updated_count = 0
    errors = []
    dataset = await get_dataset(db, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")
    
    input_type = dataset.get('input_type', 'image').upper()
    user_id = dataset.get('user_id')
    if not user_id:
        raise ValueError("Dataset has no user_id")
    
    logging.info(f"[CRUD] Found dataset: {dataset.get('name', 'Unknown')} with input_type: {input_type}")
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header
                try:
                    logging.info(f"[CRUD] Processing row {row_num}: {row}")
                    
                    # Extract common fields
                    number = row.get('#', '').strip()
                    url = row.get('URL', row.get('Link', '')).strip()  # Try URL first, then Link as fallback
                    
                    # Initialize metadata
                    metadata = {}
                    
                    # Handle different column mappings based on input_type
                    if input_type == "PDF":
                        # PDF format: use "Expected Output" column
                        expected_output = row.get('Expected Output', '').strip()
                        if not expected_output:
                            errors.append(f"Row {row_num}: Missing 'Expected Output' field")
                            logging.warning(f"[CRUD] Row {row_num}: Missing 'Expected Output' field")
                            continue
                        
                        # Add "Human Evaluation Text" to metadata if present
                        human_eval_text = row.get('Human Evaluation Text', '').strip()
                        if human_eval_text:
                            metadata['human_evaluation_text'] = human_eval_text
                            logging.info(f"[CRUD] Added human_evaluation_text to metadata: {human_eval_text[:50]}...")
                    
                    else:
                        # Image format: use "Text" column as expected_output
                        reference_text = row.get('Text', '').strip()
                        if not reference_text:
                            errors.append(f"Row {row_num}: Missing 'Text' field")
                            logging.warning(f"[CRUD] Row {row_num}: Missing 'Text' field")
                            continue
                        
                        # For images, the reference_text becomes the expected_output
                        expected_output = reference_text
                        
                        # Add "Human Evaluation Text" to metadata if present
                        human_eval_text = row.get('Human Evaluation Text', '').strip()
                        if human_eval_text:
                            metadata['human_evaluation_text'] = human_eval_text
                            logging.info(f"[CRUD] Added human_evaluation_text to metadata: {human_eval_text[:50]}...")
                    
                    if not number or not expected_output:
                        errors.append(f"Row {row_num}: Missing required fields (# or expected output)")
                        logging.warning(f"[CRUD] Row {row_num}: Missing required fields")
                        continue
                    
                    logging.info(f"[CRUD] Creating dataset file with number={number}, expected_output={expected_output[:50]}...")
                    
                    # Create new dataset file with user_id
                    file_data = DatasetFileCreate(
                        number=number,
                        url=url,
                        expected_output=expected_output,
                        file_type=input_type.lower(),  # Use dataset's input_type
                        metadata=metadata,  # Include metadata with human_evaluation_text if present
                        user_id=user_id
                    )
                    db_file = await create_dataset_file(db, file_data)
                    
                    if not db_file:
                        errors.append(f"Row {row_num}: Failed to create dataset file")
                        logging.error(f"[CRUD] Row {row_num}: Failed to create dataset file")
                        continue
                    
                    logging.info(f"[CRUD] Created dataset file with id={db_file['id']}")
                    
                    # Associate file with dataset
                    association_success = await associate_file_with_dataset(db, db_file['id'], dataset_id)
                    
                    if association_success:
                        imported_count += 1
                        logging.info(f"[CRUD] Successfully associated file_id={db_file['id']} with dataset_id={dataset_id}")
                    else:
                        errors.append(f"Row {row_num}: Failed to associate file with dataset")
                        logging.error(f"[CRUD] Row {row_num}: Failed to associate file_id={db_file['id']} with dataset_id={dataset_id}")
                        
                except Exception as e:
                    error_msg = f"Row {row_num}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(f"[CRUD] {error_msg}")
                    continue
        
        logging.info(f"[CRUD] Import completed. imported_count={imported_count}, updated_count={updated_count}, errors={len(errors)}")
        
        # Update dataset with correct file count
        await update_dataset(db, dataset_id, DatasetUpdate(
            file_count=dataset.get('file_count', 0) + imported_count,
            status=DatasetStatus.VALIDATED
        ))

        return {
            "imported_count": imported_count,
            "updated_count": updated_count,
            "errors": errors,
            "message": f"Import completed. {imported_count} new files, {updated_count} updated."
        }
    
    except FileNotFoundError:
        logging.error(f"[CRUD] CSV file not found: {csv_file_path}")
        return {
            "imported_count": 0,
            "updated_count": 0,
            "errors": [f"CSV file not found: {csv_file_path}"],
            "message": "Import failed - file not found"
        }
    except Exception as e:
        logging.error(f"[CRUD] Import error: {str(e)}")
        return {
            "imported_count": 0,
            "updated_count": 0,
            "errors": [f"Import error: {str(e)}"],
            "message": "Import failed"
        } 