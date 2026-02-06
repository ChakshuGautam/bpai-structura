from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums for better type safety
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"

class DatasetStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    ARCHIVED = "archived"

class PromptStatus(str, Enum):
    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class VersionType(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"

# Base schemas
class WordEvaluationBase(BaseModel):
    """Model for word-level evaluation results."""
    reference_word: Optional[str] = None
    transcribed_word: Optional[str] = None
    match: bool
    reason_diff: str
    word_position: int = 0  # Made optional with default value of 0
    comments: Optional[str] = None

class WordEvaluationCreate(WordEvaluationBase):
    pass

class WordEvaluation(WordEvaluationBase):
    id: int
    evaluation_id: int
    
    class Config:
        from_attributes = True

# Image schemas
class ImageBase(BaseModel):
    number: str
    url: str
    local_path: Optional[str] = None
    reference_text: str
    human_evaluation_text: str

class ImageCreate(ImageBase):
    pass

class ImageUpdate(BaseModel):
    reference_text: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    human_evaluation_text: Optional[str] = None
    expected_output: Optional[str] = None

class Image(ImageBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ImageWithEvaluations(Image):
    evaluations: List['Evaluation'] = []

# Evaluation schemas
class EvaluationBase(BaseModel):
    prompt_version_id: int
    ocr_output: Optional[str] = None
    accuracy: Optional[float] = None
    correct_words: Optional[int] = None
    total_words: Optional[int] = None
    processing_status: str = "pending"
    error_message: Optional[str] = None
    progress_percentage: Optional[int] = 0
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None

class EvaluationCreate(BaseModel):
    image_id: Optional[int] = None  # For backward compatibility
    file_id: Optional[int] = None   # New field for dataset_files
    evaluation_run_id: Optional[int] = None
    prompt_version_id: int
    force_reprocess: bool = False  # Force reprocessing even if already processed
    model_id: Optional[str] = None  # <-- Add this field
    model_info: Optional[dict] = None  # <-- Add this field

class EvaluationUpdate(BaseModel):
    ocr_output: Optional[str] = None
    accuracy: Optional[float] = None
    correct_words: Optional[int] = None
    total_words: Optional[int] = None
    processing_status: Optional[str] = None
    error_message: Optional[str] = None
    word_evaluations: Optional[List[WordEvaluationCreate]] = None
    prompt_version_id: Optional[int] = None

class Evaluation(EvaluationBase):
    id: int
    image_id: Optional[int] = None  # For backward compatibility
    file_id: Optional[int] = None   # New field for dataset_files
    created_at: datetime
    updated_at: datetime
    # Optionally include prompt_version for display
    prompt_version: Optional[str] = None
    
    class Config:
        from_attributes = True

class EvaluationWithDetails(Evaluation):
    image: Optional[Image] = None  # For backward compatibility
    file: Optional['DatasetFile'] = None  # New field for dataset_files
    word_evaluations: List[WordEvaluation] = []

# Prompt Template schemas
class PromptTemplateBase(BaseModel):
    name: str
    version: str
    prompt_text: str
    description: Optional[str] = None
    is_active: bool = False

class PromptTemplateCreate(PromptTemplateBase):
    pass

class PromptTemplateUpdate(BaseModel):
    prompt_text: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class PromptTemplate(PromptTemplateBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# CSV Import schemas
class CSVImportRequest(BaseModel):
    file_path: Optional[str] = None  # If file is already uploaded
    overwrite_existing: bool = False

class CSVImportResponse(BaseModel):
    imported_count: int
    updated_count: int
    errors: List[str] = []
    message: str

# Statistics schemas
class EvaluationStats(BaseModel):
    total_images: int
    total_evaluations: int
    pending_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_accuracy: Optional[float] = None
    accuracy_by_prompt_version: Dict[str, float] = {}

class AccuracyDistribution(BaseModel):
    high_accuracy: int  # >= 90%
    medium_accuracy: int  # 70-89%
    low_accuracy: int  # < 70%
    total_processed: int

# Batch processing schemas
class BatchProcessRequest(BaseModel):
    """Request model for batch processing evaluations"""
    image_ids: List[int]
    prompt_version: str
    force_reprocess: bool = False

class BatchProcessResponse(BaseModel):
    queued_count: int
    message: str
    job_id: Optional[str] = None  # For future job tracking

# Search and filter schemas
class ImageFilter(BaseModel):
    has_evaluations: Optional[bool] = None
    processing_status: Optional[str] = None
    prompt_version: Optional[str] = None
    accuracy_min: Optional[float] = None
    accuracy_max: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_more: bool
    
    class Config:
        from_attributes = True

class PaginatedImagesResponse(BaseModel):
    items: List[Image]
    total: int
    skip: int
    limit: int
    has_more: bool

class PaginatedEvaluationsResponse(BaseModel):
    items: List[Evaluation]
    total: int
    skip: int
    limit: int
    has_more: bool

# Progress and Status schemas
class EvaluationProgress(BaseModel):
    evaluation_id: int
    processing_status: str
    progress_percentage: int
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class EvaluationHistory(BaseModel):
    prompt_version: str
    evaluations: List[Evaluation]
    total_count: int
    avg_accuracy: Optional[float] = None
    prompt_template: Optional[PromptTemplate] = None

class PromptVersionStats(BaseModel):
    version: str
    total_evaluations: int
    successful_evaluations: int
    avg_accuracy: Optional[float] = None
    created_at: datetime
    latest_evaluation: Optional[datetime] = None

# Dataset schemas
class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: DatasetStatus = DatasetStatus.DRAFT
    input_type: str  # images, pdfs, audios
    output_type: str  # text, json
    output_schema: Optional[str] = None  # JSON schema for json output type
    input_schema: Optional[str] = None   # JSON schema for input type
    user_id: str # Clerk user ID

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[DatasetStatus] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    output_schema: Optional[str] = None
    input_schema: Optional[str] = None

class Dataset(DatasetBase):
    id: int
    file_count: int = 0  # Changed from image_count to file_count
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class DatasetWithFiles(Dataset):
    files: List['DatasetFile'] = []  # Changed from images to files

# Keep legacy DatasetWithImages for backward compatibility
class DatasetWithImages(Dataset):
    images: List['Image'] = []

# Enhanced Prompt Template schemas
class PromptFamilyBase(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = []
    user_id: str

class PromptFamilyCreate(PromptFamilyBase):
    pass

class PromptFamily(PromptFamilyBase):
    id: int
    created_at: datetime
    production_version: Optional[str] = None
    
    class Config:
        from_attributes = True

class PromptVersionBase(BaseModel):
    family_id: int
    version: str
    prompt_text: str
    changelog_message: str
    status: PromptStatus = PromptStatus.DRAFT
    user_id: str
    issues: Optional[Any] = []

class PromptVersionCreate(BaseModel):
    family_id: int
    prompt_text: str
    changelog_message: str
    version_type: VersionType
    version: Optional[str] = None
    status: PromptStatus = PromptStatus.DRAFT
    user_id: str
    issues: Optional[Any] = []

class PromptVersionUpdate(BaseModel):
    prompt_text: Optional[str] = None
    changelog_message: Optional[str] = None
    status: Optional[PromptStatus] = None
    issues: Optional[Any] = None

class PromptVersion(PromptVersionBase):
    id: int
    author: Optional[str] = None
    created_at: datetime
    last_evaluation_accuracy: Optional[float] = None
    
    class Config:
        from_attributes = True

class PromptFamilyWithVersions(PromptFamily):
    versions: List[PromptVersion] = []

# Enhanced Evaluation Run schemas
class EvaluationRunBase(BaseModel):
    name: str
    description: Optional[str] = None
    hypothesis: Optional[str] = None
    dataset_ids: List[int]
    user_id: str

class EvaluationRunCreate(EvaluationRunBase):
    prompt_configurations: List['PromptConfiguration']

class PromptConfiguration(BaseModel):
    label: str  # e.g., "Control (A)", "Variation (B)"
    family_id: int
    version_id: int

class EvaluationRunUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProcessingStatus] = None

class EvaluationRun(EvaluationRunBase):
    id: int
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress_percentage: int = 0
    current_step: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class EvaluationRunWithDetails(EvaluationRun):
    datasets: List[Dataset] = []
    prompt_configurations: List[PromptConfiguration] = []
    evaluations: List['EvaluationWithDetails'] = []
    comparison_results: Optional['ComparisonResults'] = None

# Comparison and Analysis schemas
class WordLevelComparison(BaseModel):
    image_filename: str
    word_index: int
    reference_word: str
    control_output: Optional[str] = None
    variation_output: Optional[str] = None
    status: str  # "improved", "regression", "match", "mismatch"
    error_type: Optional[str] = None

class ComparisonSummary(BaseModel):
    prompt_version: str
    label: str  # "Control (A)", "Variation (B)"
    overall_accuracy: float
    character_error_rate: float
    avg_latency_ms: int
    estimated_cost_per_1k: float
    error_breakdown: Dict[str, int]

class ComparisonResults(BaseModel):
    evaluation_run_id: int
    summary_metrics: List[ComparisonSummary]
    word_comparisons: List[WordLevelComparison]
    winner: Optional[str] = None  # Which prompt performed better
    confidence_level: Optional[float] = None

# Real-time Progress schemas
class LiveProgressUpdate(BaseModel):
    evaluation_run_id: int
    overall_progress: int
    prompt_progress: Dict[str, int]  # progress per prompt version
    current_image: Optional[str] = None
    log_entries: List[str] = []

# Historical Analysis schemas
class PerformanceTrend(BaseModel):
    prompt_version: str
    data_points: List['TrendDataPoint']
    moving_average: List[float]
    regression_alerts: List['RegressionAlert']

class TrendDataPoint(BaseModel):
    evaluation_run_id: int
    timestamp: datetime
    accuracy: float
    dataset_name: str

class RegressionAlert(BaseModel):
    detected_at: datetime
    threshold_crossed: float
    previous_average: float
    current_average: float
    severity: str  # "warning", "critical"

# API Integration schemas
class APIKey(BaseModel):
    id: int
    key_name: str
    key_preview: str  # Only show last 4 characters
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True

class APIKeyCreate(BaseModel):
    key_name: str

class APIUsageStats(BaseModel):
    api_key_id: int
    total_calls: int
    calls_today: int
    calls_this_month: int
    error_rate: float
    avg_response_time_ms: int

class APILogBase(BaseModel):
    image_url: Optional[str] = None
    ocr_output: Optional[str] = None
    prompt_version: Optional[str] = None
    user_id: Optional[str] = None
    log_metadata: Optional[Dict[str, Any]] = None

class APILogCreate(APILogBase):
    pass

class APILog(APILogBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Dataset File schemas (replacing Image schemas)
class DatasetFileBase(BaseModel):
    number: str
    url: str
    expected_output: str
    file_type: str = "image"  # image, pdf, audio, etc.
    local_path: Optional[str] = None
    metadata: Dict[str, Any] = {}  # Remove any mention of human_evaluation_text
    user_id: str  # Add user_id to base model

class DatasetFileCreate(DatasetFileBase):
    pass

class DatasetFileUpdate(BaseModel):
    number: Optional[str] = None
    url: Optional[str] = None
    expected_output: Optional[str] = None
    file_type: Optional[str] = None
    local_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DatasetFile(DatasetFileBase):
    id: int
    created_at: datetime
    updated_at: datetime
    user_id: str
    
    class Config:
        from_attributes = True

class DatasetFileWithEvaluations(DatasetFile):
    evaluations: List['Evaluation'] = []

# Dataset File Association schemas
class DatasetFileAssociationBase(BaseModel):
    dataset_id: int
    file_id: int

class DatasetFileAssociationCreate(DatasetFileAssociationBase):
    pass

class DatasetFileAssociation(DatasetFileAssociationBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# PDF Extraction schemas
class PDFExtractRequest(BaseModel):
    """Request model for PDF extraction endpoint"""
    pdf_url: str = Field(..., description="URL to the PDF file to extract from")
    prompt: str = Field(..., description="Prompt to use for extraction")

class PDFExtractResponse(BaseModel):
    """Response model for PDF extraction endpoint"""
    structured_output: Dict[str, Any] = Field(..., description="Extracted structured data from PDF")
    processing_method: str = Field(..., description="Method used for processing")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used in processing")
    success: bool = Field(..., description="Whether the extraction was successful")
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")

# Generic Evaluation schemas
class GenericEvalRequest(BaseModel):
    """Request model for generic evaluation endpoint"""
    eval_config_id: int = Field(..., description="Evaluation ID to process")
    input: Dict[str, Any] = Field(..., description="Input containing url and expected_output")

class GenericEvalResponse(BaseModel):
    """Response model for generic evaluation endpoint"""
    output: Union[Dict[str, Any], str] = Field(..., description="Output from LLM processing (can be dict or string)")
    evaluations: List[Dict[str, Any]] = Field(..., description="Word-level evaluations")
    overall_accuracy: Optional[float] = Field(None, description="Overall accuracy of the evaluation")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used in processing")
    success: bool = Field(..., description="Whether the evaluation was successful")
    error_message: Optional[str] = Field(None, description="Error message if evaluation failed") 

class DeployInferResponse(BaseModel):
    """Response model for deploy/infer endpoint (minimal fields)."""
    output: Union[Dict[str, Any], str] = Field(..., description="Output from LLM processing (can be dict or string)")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used in processing")
    success: bool = Field(..., description="Whether the evaluation was successful")
    error_message: Optional[str] = Field(None, description="Error message if evaluation failed") 