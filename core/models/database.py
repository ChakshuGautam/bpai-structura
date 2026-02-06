from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

# These are Pydantic models that mirror your Supabase tables
# They don't use SQLAlchemy but instead work with Supabase

class Dataset(BaseModel):
    id: Optional[int]
    name: str
    description: Optional[str]
    status: str = "draft"  # draft, validated, archived
    input_type: str  # images, pdfs, audios
    output_type: str  # text, json
    output_schema: Optional[str] = None  # JSON schema for json output type
    image_count: int = 0
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    last_used: Optional[datetime]
    user_id: str
    images: List["Image"] = []
    evaluation_runs: List["EvaluationRun"] = []

class PromptFamily(BaseModel):
    id: Optional[int]
    name: str
    description: Optional[str]
    tags: List[str] = []
    production_version: Optional[str]
    created_at: Optional[datetime]
    user_id: str
    versions: List["PromptVersion"] = []

class PromptVersion(BaseModel):
    id: Optional[int]
    family_id: int
    version: str
    prompt_text: str
    changelog_message: Optional[str]
    status: str = "draft"  # draft, staging, production, archived
    author: Optional[str]
    created_at: Optional[datetime]
    last_evaluation_accuracy: Optional[float]
    user_id: str
    issues: List[dict] = []
    evaluation_runs: List["EvaluationRunPrompt"] = []

class EvaluationRun(BaseModel):
    id: Optional[int]
    name: str
    description: Optional[str]
    hypothesis: Optional[str]
    status: str = "pending"  # pending, processing, success, failed
    progress_percentage: int = 0
    current_step: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]
    user_id: str
    datasets: List[Dataset] = []
    prompt_configurations: List["EvaluationRunPrompt"] = []
    evaluations: List["Evaluation"] = []

class EvaluationRunPrompt(BaseModel):
    id: Optional[int]
    evaluation_run_id: int
    prompt_version_id: int
    label: str
    evaluation_run: Optional[EvaluationRun]
    prompt_version: Optional[PromptVersion]

class APIKey(BaseModel):
    id: Optional[int]
    key_name: str
    key_hash: str
    key_preview: str
    created_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int = 0
    is_active: bool = True
    user_id: str

class Image(BaseModel):
    id: Optional[int]
    number: str
    url: Optional[str]
    local_path: Optional[str]
    reference_text: str
    human_evaluation_text: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    user_id: str
    evaluations: List["Evaluation"] = []
    datasets: List[Dataset] = []

class Evaluation(BaseModel):
    id: Optional[int]
    image_id: Optional[int] = None  # For backward compatibility
    file_id: Optional[int] = None   # New field for dataset_files
    evaluation_run_id: Optional[int]
    prompt_version_id: Optional[int] = None  # Changed from prompt_version string to ID
    ocr_output: Optional[str]
    accuracy: Optional[float]
    correct_words: Optional[int]
    total_words: Optional[int]
    processing_status: str = "pending"  # pending, processing, success, failed
    error_message: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    progress_percentage: int = 0
    current_step: Optional[str]
    estimated_completion: Optional[datetime]
    latency_ms: Optional[int]
    cost_estimate: Optional[float]
    word_evaluations_json: Optional[str]
    image: Optional[Image]
    file: Optional["DatasetFile"] = None  # New field for dataset_files
    evaluation_run: Optional[EvaluationRun]
    word_evaluations: List["WordEvaluation"] = []

class WordEvaluation(BaseModel):
    id: Optional[int]
    evaluation_id: int
    reference_word: str
    transcribed_word: Optional[str]
    match: bool
    reason_diff: str
    word_position: int
    comments: Optional[str] = None
    evaluation: Optional[Evaluation]

class DatasetFile(BaseModel):
    id: Optional[int]
    number: str
    url: str
    expected_output: str
    file_type: str = "image"  # image, pdf, audio, etc.
    local_path: Optional[str] = None
    metadata: dict = {}  # For additional fields like human_evaluation_text
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    user_id: str
    evaluations: List["Evaluation"] = []
    datasets: List[Dataset] = []

class PromptTemplate(BaseModel):
    id: Optional[int]
    name: str
    version: str
    prompt_text: str
    is_active: bool = False
    created_at: Optional[datetime]
    description: Optional[str]
    user_id: str

class APILog(BaseModel):
    id: Optional[int]
    image_url: str
    ocr_output: str
    prompt_version: str
    user_id: str
    log_metadata: dict = {}
    created_at: Optional[datetime]

# Update model references
Dataset.model_rebuild(Image=Image, EvaluationRun=EvaluationRun, DatasetFile=DatasetFile)
PromptFamily.model_rebuild(PromptVersion=PromptVersion)
PromptVersion.model_rebuild(EvaluationRunPrompt=EvaluationRunPrompt)
EvaluationRun.model_rebuild(Dataset=Dataset, EvaluationRunPrompt=EvaluationRunPrompt, Evaluation=Evaluation)
Image.model_rebuild(Evaluation=Evaluation, Dataset=Dataset)
DatasetFile.model_rebuild(Evaluation=Evaluation, Dataset=Dataset)
Evaluation.model_rebuild(Image=Image, DatasetFile=DatasetFile, EvaluationRun=EvaluationRun, WordEvaluation=WordEvaluation) 