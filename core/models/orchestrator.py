import csv
import os
import requests
import json
from datetime import datetime
import shutil
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
import aiohttp
import uuid
from .gemini_ocr import GeminiOCR

# Import our Pydantic models
from .ocr_models import (
    ProcessingStatus,
    ImageCreate,
    ImageUpdate,
    EvaluationCreate,
    EvaluationUpdate,
    WordEvaluationCreate,
    WordEvaluation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)

class OcrOrchestrator:
    """Async orchestrator for OCR evaluations."""
    
    def __init__(self):
        """Initialize the OCR orchestrator."""
        # Set up directories relative to the workspace root
        workspace_root = Path(os.getcwd())
        self.images_dir = workspace_root / "images"
        self.evaluations_dir = workspace_root / "evaluations"
        
        # Create necessary directories
        for directory in [self.images_dir, self.evaluations_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR
        try:
            logging.info("Initializing GeminiOCR...")
            self.ocr = GeminiOCR()
            logging.info("GeminiOCR initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize GeminiOCR: {str(e)}")
            raise
        
        logging.info("Initialized OcrOrchestrator")
    
    def _convert_google_drive_url(self, url: str) -> str:
        """Convert Google Drive URL to direct download link"""
        if 'drive.google.com' in url:
            # Extract file ID from Google Drive URL
            if '/file/d/' in url:
                # Format: https://drive.google.com/file/d/{file_id}/view
                file_id = url.split('/file/d/')[1].split('/')[0]
                return f"https://drive.google.com/uc?export=download&id={file_id}"
            elif '/open?id=' in url:
                # Format: https://drive.google.com/open?id={file_id}
                file_id = url.split('id=')[1].split('&')[0]
                return f"https://drive.google.com/uc?export=download&id={file_id}"
            elif 'id=' in url:
                # Format: https://drive.google.com/uc?id={file_id}
                file_id = url.split('id=')[1].split('&')[0]
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        elif 'dropbox.com' in url:
            # Convert Dropbox sharing links to direct download
            if '?dl=0' in url:
                return url.replace('?dl=0', '?dl=1')
            elif '?dl=' not in url:
                return url + '?dl=1'
        elif 'onedrive.live.com' in url or '1drv.ms' in url:
            # OneDrive links might need conversion, but this is complex
            # For now, just log that we're trying
            logging.info(f"OneDrive URL detected: {url} - may need manual conversion")
        return url
    
    def _validate_url(self, url: str) -> bool:
        """Validate if a URL is likely to be a direct file link"""
        # Check for Google Drive URLs - these are valid but need conversion
        if 'drive.google.com' in url:
            return True
            
        # Check for common file extensions in URL
        file_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
        url_lower = url.lower()
        
        # If URL has a file extension, it's likely a direct file
        if any(ext in url_lower for ext in file_extensions):
            return True
            
        # Check for common patterns that indicate HTML pages
        html_indicators = ['/view/', '/display/', '/show/', '/page/', 'index.html', 'index.php']
        if any(indicator in url_lower for indicator in html_indicators):
            logging.warning(f"URL {url} appears to be an HTML page, not a direct file link")
            return False
            
        return True
    
    async def download_image_async(self, url: str, file_id: str, dataset_input_type: Optional[str] = None) -> Optional[str]:
        """Download file asynchronously (supports both images and PDFs)"""
        try:
            # Convert Google Drive URLs to direct download links
            original_url = url
            url = self._convert_google_drive_url(url)
            if url != original_url:
                logging.info(f"Converted Google Drive URL: {original_url} -> {url}")
            
            # Validate URL before attempting download
            if not self._validate_url(url):
                logging.error(f"URL {url} appears to be invalid for file download")
                return None
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    # Log response details for debugging
                    content_type = response.headers.get('content-type', '')
                    content_length = response.headers.get('content-length', 'unknown')
                    logging.info(f"Downloading file {file_id}: content-type={content_type}, content-length={content_length}")
                    
                    # First, validate the actual content type
                    if 'text/html' in content_type.lower():
                        # Special handling for Google Drive URLs that still return HTML
                        if 'drive.google.com' in original_url:
                            logging.warning(f"Google Drive URL {original_url} still returns HTML. This might be a large file or restricted file.")
                            logging.info("Attempting to extract download link from HTML response...")
                            
                            # Try to extract the actual download link from the HTML response
                            try:
                                html_content = await response.text()
                                # Look for download links in the HTML
                                import re
                                download_patterns = [
                                    r'href="([^"]*uc\?export=download[^"]*)"',
                                    r'href="([^"]*uc\?id=[^"]*)"',
                                    r'data-download-url="([^"]*)"'
                                ]
                                
                                for pattern in download_patterns:
                                    matches = re.findall(pattern, html_content)
                                    if matches:
                                        download_url = matches[0]
                                        if download_url.startswith('/'):
                                            download_url = 'https://drive.google.com' + download_url
                                        logging.info(f"Found download link in HTML: {download_url}")
                                        
                                        # Try downloading from the extracted link
                                        async with session.get(download_url) as download_response:
                                            download_response.raise_for_status()
                                            content_type = download_response.headers.get('content-type', '')
                                            if 'text/html' not in content_type.lower():
                                                # Use the download response instead
                                                response = download_response
                                                break
                            except Exception as e:
                                logging.error(f"Failed to extract download link from HTML: {str(e)}")
                        
                        if 'text/html' in content_type.lower():
                            logging.error(f"URL {url} returns HTML content, not a valid file. Content-type: {content_type}")
                            return None
                    
                    # Determine file extension based on actual content type first, then dataset_input_type
                    if 'pdf' in content_type.lower() or url.lower().endswith('.pdf'):
                        extension = '.pdf'
                        prefix = 'pdf'
                        logging.info(f"Using PDF extension based on content-type ({content_type}) or URL ({url})")
                    elif 'image' in content_type.lower() or any(img_ext in url.lower() for img_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                        extension = '.jpg'
                        prefix = 'image'
                        logging.info(f"Using IMAGE extension based on content-type ({content_type}) or URL ({url})")
                    else:
                        # Fallback to dataset_input_type only if content type is ambiguous
                        if dataset_input_type and dataset_input_type.upper() == "PDF":
                            extension = '.pdf'
                            prefix = 'pdf'
                            logging.info(f"Using PDF extension based on dataset_input_type: {dataset_input_type}")
                        elif dataset_input_type and dataset_input_type.upper() == "IMAGE":
                            extension = '.jpg'
                            prefix = 'image'
                            logging.info(f"Using IMAGE extension based on dataset_input_type: {dataset_input_type}")
                        else:
                            # Default to image if we can't determine
                            extension = '.jpg'
                            prefix = 'image'
                            logging.info(f"Using default IMAGE extension for ambiguous content")
                    
                    # Create unique filename with UUID to avoid race conditions
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"{prefix}_{file_id}_{timestamp}_{unique_id}{extension}"
                    filepath = self.images_dir / filename
                    
                    # Write file with proper binary handling
                    with open(filepath, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    # Verify file was written correctly
                    file_size = os.path.getsize(filepath)
                    logging.info(f"Downloaded file {file_id} to {filepath} (detected as {prefix}, size: {file_size} bytes)")
                    
                    # Additional verification for PDFs
                    if extension == '.pdf':
                        # Check if file starts with PDF magic number
                        with open(filepath, 'rb') as f:
                            header = f.read(4)
                            if header != b'%PDF':
                                logging.error(f"Downloaded file {filepath} doesn't start with PDF magic number. Header: {header}")
                                # Remove the invalid file
                                os.remove(filepath)
                                return None
                            else:
                                logging.info(f"PDF file {filepath} has valid PDF header")
                    
                    # Additional validation: check if the file content matches the expected type
                    detected_content_type = self._detect_file_content_type(filepath)
                    if extension == '.pdf' and detected_content_type != 'application/pdf':
                        logging.error(f"File {filepath} has extension .pdf but content type is {detected_content_type}")
                        os.remove(filepath)
                        return None
                    elif extension == '.jpg' and not detected_content_type.startswith('image/'):
                        logging.error(f"File {filepath} has extension .jpg but content type is {detected_content_type}")
                        os.remove(filepath)
                        return None
                    
                    return str(filepath)
                    
        except Exception as e:
            logging.error(f"Failed to download file {file_id}: {str(e)}")
            return None
    
    def _detect_file_content_type(self, filepath: str) -> str:
        """Detect the actual content type of a file by examining its header"""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)  # Read first 8 bytes
                
                # Check for common file signatures
                if header.startswith(b'%PDF'):
                    return 'application/pdf'
                elif header.startswith(b'\xff\xd8\xff'):  # JPEG
                    return 'image/jpeg'
                elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                    return 'image/png'
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):  # GIF
                    return 'image/gif'
                elif header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                    return 'text/html'
                else:
                    return 'unknown'
        except Exception as e:
            logging.error(f"Error detecting content type for {filepath}: {str(e)}")
            return 'unknown'
    
    def _validate_pdf_file(self, filepath: str) -> bool:
        """Validate if a file is a proper PDF by checking its header"""
        try:
            if not os.path.exists(filepath):
                logging.error(f"PDF file {filepath} does not exist")
                return False
                
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logging.error(f"PDF file {filepath} is empty (0 bytes)")
                return False
                
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    # Read a bit more to help with debugging
                    f.seek(0)
                    first_bytes = f.read(20)
                    logging.error(f"PDF file {filepath} doesn't start with PDF magic number. First 20 bytes: {first_bytes}")
                    return False
                else:
                    logging.info(f"PDF file {filepath} has valid PDF header")
                    return True
        except Exception as e:
            logging.error(f"Error validating PDF file {filepath}: {str(e)}")
            return False
    
    async def process_single_evaluation(self, image_url: str, reference_text: str, image_number: str, prompt_text: Optional[str] = None, file_id: Optional[int] = None, dataset_input_type: Optional[str] = None, model_id: Optional[str] = None, model_info: Optional[dict] = None) -> Dict:
        """Process a single file evaluation asynchronously based on dataset input_type"""
        local_file_path = None
        try:
            logging.info(f"Processing evaluation for file {image_number} with input_type: {dataset_input_type} and model_id: {model_id}")
            # Download file
            local_file_path = await self.download_image_async(image_url, image_number, dataset_input_type)
            if not local_file_path:
                return {
                    'success': False,
                    'error': 'Failed to download file'
                }
            # Determine model to use
            model_to_use = model_id or 'gemini-2.5-flash'
            # Determine processing method based on dataset input_type
            if dataset_input_type == "PDF":
                # Use PDF processing
                logging.info(f"Processing {image_number} as PDF with model {model_to_use}")
                # Validate PDF file before processing
                if not self._validate_pdf_file(local_file_path):
                    logging.error(f"Invalid PDF file detected: {local_file_path}")
                    return {
                        'success': False,
                        'error': f'Invalid or corrupted PDF file: {local_file_path}'
                    }
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ocr.extract_from_pdf_with_model(local_file_path, prompt_text or '', model_to_use)
                )
            else:
                # Default to image processing (for "images" or any other type)
                logging.info(f"Processing {image_number} as image with model {model_to_use}")
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ocr.extract_text_with_model(local_file_path, reference_text, prompt_text, model_to_use)
                )
            if not result:
                return {
                    'success': False,
                    'error': 'Processing returned no result'
                }
            # Handle different result formats
            if dataset_input_type == "PDF":
                # PDF processing returns structured data
                evaluation_dict = self._create_pdf_evaluation_dict(result, file_id, image_number)
            else:
                # Image processing returns word evaluations
                evaluation_dict = self._create_image_evaluation_dict(result, file_id, image_number)
            # Return evaluation data
            return {
                'success': True,
                'evaluation': evaluation_dict,
                'local_file_path': local_file_path,
                'tokens_used': result.get('tokens_used', 0)
            }
        except Exception as e:
            logging.error(f"Error processing evaluation for file {image_number}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if local_file_path and os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                    logging.info(f"Cleaned up temp file: {local_file_path}")
                except Exception as cleanup_err:
                    logging.error(f"Failed to delete temp file {local_file_path}: {cleanup_err}")
    
    async def _process_image_file(self, local_file_path: str, reference_text: str, prompt_text: Optional[str] = None) -> Dict:
        """Process an image file using extract_text_with_model method"""
        loop = asyncio.get_event_loop()
        # Use the new extract_text_with_model function with default model
        return await loop.run_in_executor(
            None,
            self.ocr.extract_text_with_model,
            local_file_path,
            reference_text,
            prompt_text,
            'gemini-2.5-flash'  # Default model
        )
        
        # Original code (commented out):
        # return await loop.run_in_executor(
        #     None, 
        #     self.ocr.extract_text, 
        #     local_file_path, 
        #     reference_text,
        #     prompt_text
        # )
    
    async def _process_pdf_file(self, local_file_path: str, reference_text: str, prompt_text: Optional[str] = None) -> Dict:
        """Process a PDF file using extract_from_pdf method"""
        # For PDFs, we need to create a prompt that includes the expected structure
        if not prompt_text:
            # Create a default prompt for PDF processing
            prompt_text = f"""
Extract information from this PDF and return it in a structured format.
Reference text: {reference_text}

Please extract the key information and return it in JSON format.
"""
        
        loop = asyncio.get_event_loop()
        
        # Extract structured information from PDF
        extraction_result = await loop.run_in_executor(
            None,
            self.ocr.extract_from_pdf_with_model,
            local_file_path,
            prompt_text,
            'gemini-2.5-flash'
        )
        
        # Try to parse reference_text as JSON to get expected structure
        expected_structure = None
        
        # Add debugging to understand what reference_text contains
        # logging.info(f"Processing PDF with reference_text (type: {type(reference_text)}, length: {len(reference_text) if reference_text else 0})")
        logging.info(f"Reference text preview: {repr(reference_text[:200]) if reference_text else 'None'}")
        
        try:
            if reference_text:
                # Try to parse as JSON regardless of starting character
                # First, try to clean the string
                cleaned_text = reference_text.strip()
                
                # If it doesn't start with {, try to find JSON within the text
                if not cleaned_text.startswith('{'):
                    # Look for JSON object in the text
                    import re
                    json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                    if json_match:
                        cleaned_text = json_match.group(0)
                        logging.info(f"Found JSON object in reference_text: {cleaned_text[:100]}...")
                    else:
                        logging.info("No JSON object found in reference_text")
                        cleaned_text = None
                
                if cleaned_text and cleaned_text.startswith('{'):
                    expected_structure = json.loads(cleaned_text)
                    # logging.info(f"Successfully parsed expected structure from reference_text: {expected_structure}")
                else:
                    logging.info("Reference text does not contain valid JSON structure")
            else:
                logging.info("Reference text is None or empty")
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse reference_text as JSON: {e}")
            logging.error(f"Reference text that failed to parse: {repr(reference_text[:500])}")
        except Exception as e:
            logging.error(f"Unexpected error parsing reference_text: {e}")
        
        # If we have an expected structure and successful extraction, perform comparison
        if expected_structure and extraction_result.get('structured_output') and 'error' not in extraction_result:
            logging.info("Performing JSON structure comparison (keywise)")
            try:
                # comparison_result = await loop.run_in_executor(
                #     None,
                #     self.ocr.compare_json_structures,
                #     expected_structure,
                #     extraction_result['structured_output']
                # )
                # Switched to keywise comparison (can revert to above if needed)
                comparison_result = await loop.run_in_executor(
                    None,
                    self.ocr.compare_json_structures_keywise,
                    expected_structure,
                    extraction_result['structured_output']
                )
                extraction_result['comparison'] = comparison_result
                logging.info(f"Comparison completed: {comparison_result}")
            except Exception as e:
                logging.error(f"Error during JSON structure comparison: {e}")
                extraction_result['comparison'] = None
        else:
            logging.info("Skipping comparison - missing expected_structure or extraction_result")
        
        return extraction_result
    
    def _create_image_evaluation_dict(self, result: Dict, file_id: Optional[int], image_number: str) -> Dict:
        """Create evaluation dict for image processing results"""
        # Create Pydantic models for the response
        word_evaluations = [
            WordEvaluationCreate(
                reference_word=eval['reference_word'],
                transcribed_word=eval['transcribed_word'],
                match=eval['match'],
                reason_diff=eval['reason_diff'],
                word_position=eval.get('word_position', 0)
            ) for eval in result.get('evaluations', [])
        ]

        logging.info(f"word_evaluations for image {image_number}: {[we.dict() for we in word_evaluations]}")

        return {
            "file_id": file_id if file_id is not None else None,
            "image_id": None if file_id is not None else image_number,
            "prompt_version_id": None,
            "ocr_output": result.get('full_text', ''),
            "accuracy": result.get('accuracy', 0),
            "correct_words": result.get('correct_words', 0),
            "total_words": result.get('total_words', 0),
            "processing_status": ProcessingStatus.SUCCESS,
            "word_evaluations": [we.dict() for we in word_evaluations]
        }
    
    def _create_pdf_evaluation_dict(self, result: Dict, file_id: Optional[int], image_number: str) -> Dict:
        """Create evaluation dict for PDF processing results"""
        # For PDFs, we store the structured output and create a summary
        structured_output = result.get('structured_output', {})
        comparison_result = result.get('comparison', {})
        
        # Create word evaluations based on comparison results
        word_evaluations = []
        
        if comparison_result and 'key_comparisons' in comparison_result:
            # Create word evaluations for each key in the comparison
            for key, comparison in comparison_result['key_comparisons'].items():
                expected_value = comparison.get('expected_value', '')
                actual_value = comparison.get('actual_value', '')
                similarity = comparison.get('similarity', 0)
                explanation = comparison.get('explanation', '')
                
                # Convert values to strings for storage, including the key
                expected_json = json.dumps({key: expected_value}, ensure_ascii=False) if expected_value else json.dumps({key: ""}, ensure_ascii=False)
                actual_json = json.dumps({key: actual_value}, ensure_ascii=False) if actual_value else json.dumps({key: ""}, ensure_ascii=False)
                
                # Determine if it's a match based on similarity threshold
                is_match = similarity >= 0.8  # 80% similarity threshold
                
                word_evaluations.append(
                    WordEvaluationCreate(
                        reference_word=expected_json,
                        transcribed_word=actual_json,
                        match=is_match,
                        reason_diff=f"Similarity: {similarity:.2f}. {explanation}",
                        word_position=len(word_evaluations)  # Use position as index
                    )
                )
            
            # Add evaluations for missing keys
            for missing_key in comparison_result.get('missing_keys', []):
                expected_json = json.dumps({missing_key: "expected_value"}, ensure_ascii=False)
                actual_json = json.dumps({missing_key: "MISSING"}, ensure_ascii=False)
                
                word_evaluations.append(
                    WordEvaluationCreate(
                        reference_word=expected_json,
                        transcribed_word=actual_json,
                        match=False,
                        reason_diff=f"Key '{missing_key}' was expected but not found in output",
                        word_position=len(word_evaluations)
                    )
                )
            
            # Add evaluations for extra keys
            for extra_key in comparison_result.get('extra_keys', []):
                expected_json = json.dumps({"NOT_EXPECTED": "key_not_in_expected"}, ensure_ascii=False)
                actual_json = json.dumps({extra_key: "actual_value"}, ensure_ascii=False)
                
                word_evaluations.append(
                    WordEvaluationCreate(
                        reference_word=expected_json,
                        transcribed_word=actual_json,
                        match=False,
                        reason_diff=f"Key '{extra_key}' was found in output but not expected",
                        word_position=len(word_evaluations)
                    )
                )
        else:
            # Fallback: create a simple word evaluation for PDFs (since they don't have word-level evaluation)
            # This maintains compatibility with the existing evaluation structure
            word_evaluations = [
                WordEvaluationCreate(
                    reference_word="PDF Content",
                    transcribed_word="Dummy Output",
                    match=True,  # PDFs are considered successful if they return structured data
                    reason_diff="PDF processed successfully",
                    word_position=0
                )
            ]

        # Calculate accuracy based on comparison results
        if comparison_result and 'overall_similarity' in comparison_result:
            accuracy = comparison_result['overall_similarity'] * 100
            correct_words = sum(1 for eval in word_evaluations if eval.match)
            total_words = len(word_evaluations)
        else:
            accuracy = 100.0  # PDFs get 100% accuracy if they return structured data
            correct_words = 1
            total_words = 1

        return {
            "file_id": file_id if file_id is not None else None,
            "image_id": None if file_id is not None else image_number,
            "prompt_version_id": None,
            "ocr_output": json.dumps(structured_output, ensure_ascii=False, indent=2),  # Store structured output as OCR output
            "accuracy": accuracy,
            "correct_words": correct_words,
            "total_words": total_words,
            "processing_status": ProcessingStatus.SUCCESS,
            "word_evaluations": [we.dict() for we in word_evaluations],
            "structured_output": structured_output,  # Additional field for PDF results
            "comparison_result": comparison_result  # Additional field for comparison results
        }

# Maintain backward compatibility with existing ImageProcessor
class ImageProcessor:
    """Orchestrator for processing images from a CSV file using Gemini OCR."""
    
    def __init__(self, csv_path: str, max_retries: int = 3):
        """
        Initialize the image processor.
        
        Args:
            csv_path: Path to the CSV file containing image URLs
            max_retries: Maximum number of retries for failed processing
        """
        self.csv_path = csv_path
        self.max_retries = max_retries
        
        # Set up directories relative to the workspace root
        workspace_root = Path(os.getcwd())
        self.images_dir = workspace_root / "images"
        self.evaluations_dir = workspace_root / "evaluations"
        self.failed_dir = workspace_root / "failed"
        
        # Create necessary directories
        for directory in [self.images_dir, self.evaluations_dir, self.failed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR
        self.ocr = GeminiOCR()
        
        # Track processing statistics
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0
        }
        
        # Track failed entries for retry
        self.failed_entries: List[Dict] = []
        
        logging.info(f"Initialized ImageProcessor with CSV: {csv_path}")
        logging.info(f"Images directory: {self.images_dir}")
        logging.info(f"Evaluations directory: {self.evaluations_dir}")
        logging.info(f"Max retries: {self.max_retries}")
    
    def download_image(self, url: str, image_id: str) -> Optional[str]:
        """Download image and save with unique name"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create unique filename using image ID and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{image_id}_{timestamp}.jpg"
            filepath = self.images_dir / filename
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"Successfully downloaded image {image_id} to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logging.error(f"Failed to download image {image_id}: {str(e)}")
            return None
    
    def save_evaluation_json(self, image_number: str, image_url: str, reference_text: str, 
                           transcribed_text: str, evaluations: List[Dict], local_image_path: str) -> str:
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{image_number}_{timestamp}.json"
        filepath = self.evaluations_dir / filename
        
        # Calculate accuracy metrics
        total_words = len(evaluations)
        correct_words = sum(1 for eval in evaluations if eval['match'])
        accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
        
        data = {
            "image_info": {
                "number": image_number,
                "url": image_url,
                "reference_text": reference_text,
                "timestamp": timestamp,
                "local_image_path": str(Path(local_image_path).relative_to(self.images_dir))
            },
            "evaluation": {
                "full_text": transcribed_text,
                "word_evaluations": evaluations,
                "metrics": {
                    "total_words": total_words,
                    "correct_words": correct_words,
                    "accuracy": accuracy
                }
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved evaluation for image {image_number} to {filepath}")
        logging.info(f"Evaluation data being saved: {json.dumps(data, ensure_ascii=False)[:1000]}")
        return filename
    
    def process_single_image(self, row: Dict, retry_count: int = 0) -> Tuple[bool, Dict]:
        """Process a single image with retry logic"""
        image_number = row['#']
        image_url = row['Link']
        reference_text = row['Text']
        
        logging.info(f"Processing image {image_number} (attempt {retry_count + 1}/{self.max_retries})")
        
        try:
            # Download image
            local_image_path = self.download_image(image_url, image_number)
            if not local_image_path:
                raise Exception("Failed to download image")
            
            # Process image using the new extract_text_with_model function
            result = self.ocr.extract_text_with_model(local_image_path, reference_text, None, "claude-3-sonnet-20240229")
            if not result:
                raise Exception("OCR returned no result")
            
            # Original code (commented out):
            # result = self.ocr.extract_text(local_image_path, reference_text)
            # if not result:
            #     raise Exception("OCR returned no result")
            
            # Save evaluation
            evaluation_file = self.save_evaluation_json(
                image_number, image_url, reference_text,
                result['full_text'], result['evaluations'],
                local_image_path
            )
            
            # Update row with results
            # Escape newlines in OCR output to prevent multi-row CSV issues
            ocr_output = result['full_text'].replace('\n', '\\n').replace('\r', '\\r') if result['full_text'] else ''
            row['OCR Output (Gemini - Flash)'] = ocr_output
            row['Word Evaluations'] = json.dumps(result['evaluations'], ensure_ascii=False)
            row['Accuracy'] = f"{result['accuracy']:.2f}%"
            row['Correct Words'] = result['correct_words']
            row['Total Words'] = result['total_words']
            row['Evaluation JSON'] = evaluation_file
            row['Local Image'] = str(Path(local_image_path).relative_to(self.images_dir))
            
            logging.info(f"Successfully processed image {image_number}")
            return True, row
            
        except Exception as e:
            logging.error(f"Error processing image {image_number}: {str(e)}")
            
            if retry_count < self.max_retries - 1:
                logging.info(f"Retrying image {image_number} (attempt {retry_count + 2})")
                return self.process_single_image(row, retry_count + 1)
            else:
                # Move failed image to failed directory if it exists
                if 'local_image_path' in locals() and os.path.exists(local_image_path):
                    failed_path = self.failed_dir / Path(local_image_path).name
                    shutil.move(local_image_path, failed_path)
                    row['Local Image'] = str(failed_path.relative_to(self.images_dir))
                
                row['Processing Status'] = f"Failed after {self.max_retries} attempts: {str(e)}"
                return False, row
    
    def process_csv(self):
        """Process all images in CSV with retry logic"""
        logging.info("Starting CSV processing")
        
        # Read CSV
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.stats['total'] = len(rows)
        logging.info(f"Found {self.stats['total']} images to process")
        
        # Process each row
        processed_rows = []
        for row in rows:
            success, processed_row = self.process_single_image(row)
            processed_rows.append(processed_row)
            
            if success:
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
                self.failed_entries.append(processed_row)
        
        # Save updated CSV
        output_csv = f"{os.path.splitext(self.csv_path)[0]}_updated.csv"
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=processed_rows[0].keys())
            writer.writeheader()
            writer.writerows(processed_rows)
        
        # Save failed entries for retry
        if self.failed_entries:
            failed_csv = f"{os.path.splitext(self.csv_path)[0]}_failed.csv"
            with open(failed_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.failed_entries[0].keys())
                writer.writeheader()
                writer.writerows(self.failed_entries)
        
        # Log summary
        logging.info("\nProcessing Summary:")
        logging.info(f"Total images: {self.stats['total']}")
        logging.info(f"Successfully processed: {self.stats['successful']}")
        logging.info(f"Failed: {self.stats['failed']}")
        logging.info(f"Total retries: {self.stats['retries']}")
        
        if self.failed_entries:
            logging.info(f"\nFailed entries saved to: {failed_csv}")
            logging.info("Failed entries can be retried by running the script again with the failed CSV")
        
        logging.info(f"\nUpdated CSV saved to: {output_csv}")
        logging.info(f"Evaluations saved to: {self.evaluations_dir}")
        logging.info(f"Images saved to: {self.images_dir}")
        if self.failed_entries:
            logging.info(f"Failed images saved to: {self.failed_dir}")

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.orchestrator <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    processor = ImageProcessor(csv_path)
    processor.process_csv()

if __name__ == "__main__":
    main() 