import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from io import BytesIO
from pydantic import BaseModel, Field
import logging
import re
from .transformation_nl import transform_word_evaluations_nl
# # PDF processing imports
# try:
#     import fitz  # PyMuPDF
#     import pdfplumber
#     from pdf2image import convert_from_path
#     PDF_AVAILABLE = True
# except ImportError:
#     PDF_AVAILABLE = False
#     logging.warning("PDF processing libraries not available. Install PyMuPDF, pdfplumber, and pdf2image for PDF support.")

# Load environment variables
load_dotenv()

class WordEvaluation(BaseModel):
    """Model for word-level evaluation results."""
    reference_word: Optional[str] = None
    transcribed_word: Optional[str] = None
    match: bool
    reason_diff: str
    comments: Optional[str] = None

class GeminiOCR:
    """A class to handle OCR operations using Google's Gemini API."""
    
    def __init__(self, timeout: int = 60):
        """
        Initialize the Gemini OCR client.
        
        Args:
            timeout: Timeout in seconds for API calls
        """
        self.timeout = timeout
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            error_msg = "GOOGLE_API_KEY environment variable is not set"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info("Initializing Gemini client with API key...")
        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout * 1000}
        )
        logging.info("Gemini client initialized successfully")
    
    def _process_image(self, image: Union[PIL.Image.Image, str, Path]) -> types.Part:
        """
        Process an image into a format suitable for the Gemini API.
        
        Args:
            image: Can be a PIL Image, file path (str or Path)
            
        Returns:
            Processed image part for Gemini API
        """
        if isinstance(image, PIL.Image.Image):
            # Handle PIL Image - convert to bytes
            image_bytes = BytesIO()
            image.save(image_bytes, format="WEBP")
            return types.Part.from_bytes(
                data=image_bytes.getvalue(),
                mime_type="image/webp"
            )
        elif isinstance(image, (str, Path)):
            # Handle file path
            file_path = str(image)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Upload file to Gemini
            return self.client.files.upload(file=file_path)
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Supported types: PIL.Image.Image, str, Path"
            )
    
    def _is_valid_word_evaluation_format(self, output: str) -> bool:
        """
        Validate that output is in expected word evaluation format.

        Returns False if:
        - Output contains nested JSON with "accuracy" and "full_text" keys
        - Output contains markdown formatting like ** or ```
        - Output doesn't start with [ (expected array)
        - Output contains "Transcribe:" or other instruction text
        """
        if not output:
            return False

        output_stripped = output.strip()

        # Check for invalid patterns
        invalid_patterns = [
            '"accuracy"',
            '"full_text"',
            '**',
            '```',
            'Transcribe:',
            'Based on the provided',
            'handwritten',
        ]

        for pattern in invalid_patterns:
            if pattern in output_stripped:
                logging.warning(f"[GEMINI_OCR] Found invalid pattern: {pattern}")
                return False

        # Should start with [ for JSON array
        if not output_stripped.startswith('['):
            logging.warning(f"[GEMINI_OCR] Output doesn't start with '[': {output_stripped[:50]}")
            return False

        # Try to parse and check structure
        try:
            data = json.loads(output_stripped)
            if not isinstance(data, list):
                logging.warning(f"[GEMINI_OCR] Output is not a list")
                return False

            if len(data) == 0:
                logging.warning(f"[GEMINI_OCR] Output is empty array")
                return False

            # Check first item has expected keys
            first_item = data[0]
            required_keys = ['reference_word', 'transcribed_word', 'match']
            for key in required_keys:
                if key not in first_item:
                    logging.warning(f"[GEMINI_OCR] Missing required key: {key}")
                    return False

            return True
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return False

    def _process_pdf_simple(self, pdf_path: Union[str, Path]) -> types.Part:
        """
        Process a PDF by directly uploading it to Gemini.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDF part for Gemini API
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Directly upload PDF to Gemini - no preprocessing needed!
        return self.client.files.upload(file=pdf_path)

    def extract_text(
        self,
        image: Union[PIL.Image.Image, str, Path],
        reference_text: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> Dict:
        """
        Extract text from an image using Gemini API and evaluate against reference text if provided.
        
        Args:
            image: The image to process (PIL Image or file path)
            reference_text: Optional reference text to compare against
            prompt_text: Optional custom prompt to use for the LLM call
            
        Returns:
            Dict containing the extracted text and evaluation results
        """
        # Process the image
        media_part = self._process_image(image)

        # Use the provided prompt_text if given, else fallback to the hardcoded prompt
        prompt_structure = """
**Input Provided to You:**
*   An image containing the handwritten Hindi text.
*   A string containing the reference Hindi text (this is the ground truth the student was asked to write).

**CRITICAL OUTPUT RULES - READ CAREFULLY:**
1. You MUST ALWAYS return a valid JSON array starting with [ and ending with ]
2. NEVER return English explanations, descriptions, or reasoning outside the JSON array
3. NEVER use markdown formatting like ** or ```
4. NEVER include keys like "accuracy" or "full_text" in the output
5. NEVER return text like "I have carefully examined", "Based on the provided image", "The image appears to be", etc.

**Special Cases You MUST Handle:**
*   If the image is BLANK or contains no text: Return the JSON array with each transcribed_word set to null
*   If the image contains WRONG CONTENT (math, objects, etc.): Return the JSON array with transcribed_word set to null for each reference word
*   If text is COMPLETELY ILLEGIBLE: Return the JSON array with transcribed_word set to "[illegible]"
*   If you CANNOT find Hindi text: Still return the JSON array, don't explain in English

**Output Format (Mandatory):**
You must produce a JSON list of objects. Each object in the list represents the evaluation of a single word from the reference text, in the order they appear. Each object must contain the following keys:

*   `reference_word` (string): The word from the reference text.
*   `transcribed_word` (string/null): The corresponding word or segment transcribed from the image.
    *   If a directly corresponding word is found, provide it.
    *   If the word seems to be part of a merged segment in the transcription (e.g., reference "मत कर" transcribed as "मतकर"), this field might show the merged segment for both reference words involved.
    *   If the word from the reference is entirely missing in the transcription, use `null` or an empty string for this field.
    *   If a word in the image is completely illegible, you can represent it as `"[illegible]"`.
*   `match` (boolean): `true` if the `transcribed_word` (or the relevant part of it) is an exact character-by-character match with the `reference_word` (including all matras and conjunct characters). `false` otherwise.
*   `reason_diff` (string):
    *   If `match` is `true`, this field can be an empty string or a brief confirmation like "Exact match."
    *   If `match` is `false`, provide a concise explanation of the mismatch. Examples include:
        *   "Spelling error: Transcribed '[transcribed]' vs reference '[reference]' (e.g., incorrect matra, different character)."
        *   "Missing matra: e.g., 'ा' missing in '[transcribed]'."
        *   "Extra character: e.g., additional 'र्' in '[transcribed]'."
        *   "Word missing: Reference word '[reference]' not found in transcription at this position."
        *   "Segmentation error: Reference '[reference]' appears merged in transcription (e.g., as part of '[merged_transcribed_segment]')."
        *   "Segmentation error: Reference '[reference]' appears split in transcription."
        *   "Illegible word in transcription."

**Detailed Instructions for Comparison and Evaluation:**
*   **Sequential Evaluation:** Iterate through the words of the reference text in order. For each `reference_word`, identify its corresponding counterpart(s) or absence in your transcribed text.
*   **Accuracy:** The comparison must be exact. Differences in matras (vowel signs), anusvara, visarga, chandrabindu, and base characters constitute a mismatch.
*   **Word Segmentation:**
    *   If the student merges words that are separate in the reference (e.g., reference "मत कर", transcribed "मतकर"), then for `reference_word: "मत"`, the `transcribed_word` could be "मतकर", `match: false`, and `reason_diff` should explain the merge. Similarly for `reference_word: "कर"`.
    *   If the student splits a word that is single in the reference, adapt the `reason_diff` accordingly.
*   **Missing/Extra Words:**
    *   If a reference word is missing from the transcription, indicate this clearly.
    *   If the transcription contains extra words not present in the reference text, these should ideally be noted after all reference words have been evaluated, perhaps as additional entries with `reference_word: null` or by detailing them in the `reason_diff` of a nearby word if they disrupt the alignment significantly. For simplicity, prioritize evaluating against the reference words first.

**Example (Conceptual):**
If Reference Text is: `हर पल`
And Transcribed Text from image is: `हर पल`
Output:
```json
[
  {
    "reference_word": "हर",
    "transcribed_word": "हर",
    "match": true,
    "reason_diff": "Exact match."
  },
  {
    "reference_word": "पल",
    "transcribed_word": "पल",
    "match": true,
    "reason_diff": "Exact match."
  }
]
```

If Reference Text is: लड़ाई
And Transcribed Text from image is: लड़ई
Output:
```json
[
  {
    "reference_word": "लड़ाई",
    "transcribed_word": "लड़ई",
    "match": false,
    "reason_diff": "Spelling error: Transcribed 'लड़ई' is missing the 'ा' (aa) matra found in 'लड़ाई'."
  }
]
```

If Reference Text is: उस तट पर
And Transcribed Text from image is: उस पर (student missed "तट")
Output:
```json
[
  {
    "reference_word": "उस",
    "transcribed_word": "उस",
    "match": true,
    "reason_diff": "Exact match."
  },
  {
    "reference_word": "तट",
    "transcribed_word": null,
    "match": false,
    "reason_diff": "Word missing: Reference word 'तट' not found in transcription at this position."
  },
  {
    "reference_word": "पर",
    "transcribed_word": "पर",
    "match": true,
    "reason_diff": "Exact match."
  }
]
```

Begin by transcribing the provided image, then proceed to the word-by-word evaluation against the reference text, structuring your final output strictly in the JSON format specified.
"""

        sample_prompt_text = """
You are an AI assistant specialized in Optical Character Recognition (OCR) and text comparison for handwritten Hindi. You will be provided with an image containing handwritten Hindi text and a corresponding reference Hindi text that the handwriting is supposed to match.

Your task is to:
1.  **Transcribe:** Accurately transcribe the Hindi words from the provided image. Focus exclusively on Hindi script and words. Ignore any non-Hindi elements.
2.  **Tokenize:** Internally, split both the reference text and your transcribed text into individual words. Word boundaries are typically defined by spaces.
3.  **Compare and Evaluate:** Perform a word-by-word comparison of your transcribed text against the reference text. Your output should be a detailed evaluation for each word based on the sequence in the reference text.
"""
        
        # logging.info(f"[OCR] Prompt text used for evaluation : {prompt_text}")
        # logging.info(f"Prompt_Text : {prompt_text}")
        if prompt_text:
            # prompt = f"{prompt_text}\n{prompt_structure}"
            prompt = prompt_text
        else:
            prompt = f"{sample_prompt_text}\n{prompt_structure}"
        # logging.info(f"[OCR] Prompt used for evaluation : {prompt}")
        

        logging.info(f"[GEMINI OCR] Reference Text: {reference_text}")
        if reference_text:
            prompt = f"Reference Text: {reference_text}\n\n{prompt}"
        
        max_retries = 3
        retry_count = 0

        # Different temperatures for each retry attempt
        # Start conservative (0), then increase for variety
        temperatures = [0, 0.3, 0.5]

        while retry_count < max_retries:
            try:
                # Prepare configuration with varying temperature
                temperature = temperatures[retry_count]
                config = {
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                }

                logging.info(f"[GEMINI_OCR] Attempt {retry_count + 1}/{max_retries} with temperature={temperature}")

                # Make the API call
                response = self.client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=[media_part, prompt],
                    config=config,
                )

                # Extract and parse the response
                output = response.candidates[0].content.parts[0].text
                logging.info(f"[GEMINI_OCR] Raw output (attempt {retry_count + 1}): {output[:200]}")

                # Validate format before parsing
                if not self._is_valid_word_evaluation_format(output):
                    logging.warning(f"[GEMINI_OCR] Invalid format detected on attempt {retry_count + 1}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.info(f"[GEMINI_OCR] Retrying... (attempt {retry_count + 1}/{max_retries})")
                        continue
                    else:
                        logging.error(f"[GEMINI_OCR] Max retries reached. Returning error.")
                        return {"error": "Invalid output format after retries", "raw_output": output[:500]}

                evaluations = json.loads(output)
                logging.info(f"[GEMINI_OCR] Successfully parsed JSON with {len(evaluations)} evaluations")

                # Validate evaluations against our schema
                evaluations = transform_word_evaluations_nl(evaluations)
                word_evaluations = [WordEvaluation(**eval_data) for eval_data in evaluations]
                break  # Success! Exit retry loop

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.warning(f"[GEMINI_OCR] Parsing error on attempt {retry_count + 1}: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"[GEMINI_OCR] Max retries reached after parsing errors")
                    return {"error": f"Failed to parse response: {str(e)}", "raw_output": output[:500]}
                logging.info(f"[GEMINI_OCR] Retrying... (attempt {retry_count + 1}/{max_retries})")
                continue

        # Continue with successful result
        try:
            
            # Construct the full text from transcribed words
            transcribed_words = []
            for eval in word_evaluations:
                if eval.transcribed_word and eval.transcribed_word != "[illegible]":
                    transcribed_words.append(eval.transcribed_word)
            
            full_text = " ".join(transcribed_words)
            
            # Calculate accuracy metrics
            total_words = len(word_evaluations)
            correct_words = sum(1 for eval in word_evaluations if eval.match)
            accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
            
            # Get token usage
            total_tokens = 0
            if response.usage_metadata:
                total_tokens = response.usage_metadata.total_token_count
            
            result_dict = {
                "full_text": full_text,
                "evaluations": [eval.dict() for eval in word_evaluations],
                "accuracy": accuracy,
                "correct_words": correct_words,
                "total_words": total_words,
                "tokens_used": total_tokens
            }
            # logging.info(f"Final result dict to return: {json.dumps(result_dict, ensure_ascii=False)[:1000]}")
            return result_dict
            
        except APIError as e:
            print(f"Error calling Gemini API: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"error": str(e)}
    
    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        prompt_text: str,
    ) -> Dict:
        """
        Extract structured information from a PDF using Gemini API.
        
        Args:
            pdf_path: Path to the PDF file
            prompt_text: Prompt that includes the expected JSON structure
            
        Returns:
            Dict containing the extracted structured information
        """
        # Process the PDF - much simpler now!
        media_part = self._process_pdf_simple(pdf_path)
        
        try:
            # Prepare configuration
            config = {
                "temperature": 0,
                "response_mime_type": "application/json",
            }
            
            # Combine PDF with the prompt
            contents = [media_part, prompt_text]
            
            # Make the API call
            response = self.client.models.generate_content(
                model="gemini-1.5-pro",
                contents=contents,
                config=config,
            )
            logging.info(f"[GEMINI OCR]Response : {response}")
            # Extract and parse the response
            output = response.candidates[0].content.parts[0].text
            structured_output = json.loads(output)
            logging.info(f"[GEMINI_OCR] Structured output: {structured_output}")
            # Get token usage
            total_tokens = 0
            if response.usage_metadata:
                total_tokens = response.usage_metadata.total_token_count
            
            result_dict = {
                "structured_output": structured_output,
                "tokens_used": total_tokens,
                "processing_method": "direct_pdf_upload"
            }
            
            return result_dict
            
        except APIError as e:
            logging.error(f"Error calling Gemini API: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"error": str(e)}
    
    def _process_pdf_for_custom_api(self, pdf_path: Union[str, Path]) -> str:
        """
        Process a PDF for the custom API endpoint.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Data URI string for the PDF
        """
        import base64
        
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Read the PDF file and encode as base64
        with open(pdf_path, "rb") as pdf_file:
            encoded_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        
        # Create data URI
        pdf_data_uri = f"data:application/pdf;base64,{encoded_pdf}"
        
        return pdf_data_uri

    def _process_image_for_custom_api(self, image: Union[PIL.Image.Image, str, Path]) -> str:
        """
        Process an image for the custom API endpoint.
        
        Args:
            image: Can be a PIL Image, file path (str or Path)
            
        Returns:
            Data URI string for the image
        """
        import base64
        
        if isinstance(image, PIL.Image.Image):
            # Handle PIL Image - convert to base64
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            encoded_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            image_data_uri = f"data:image/png;base64,{encoded_image}"
        elif isinstance(image, (str, Path)):
            # Handle file path
            file_path = str(image)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            # Read the image file and encode as base64
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Determine MIME type based on file extension
            file_extension = Path(file_path).suffix.lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff'
            }
            mime_type = mime_type_map.get(file_extension, 'image/jpeg')
            image_data_uri = f"data:{mime_type};base64,{encoded_image}"
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Supported types: PIL.Image.Image, str, Path"
            )
        
        return image_data_uri

    def extract_from_pdf_with_model(
        self,
        pdf_path: Union[str, Path],
        prompt_text: str,
        model: str = "gemini-2.5-flash",
    ) -> Dict:
        """
        Extract structured information from a PDF using the specified API endpoint.
        
        Args:
            pdf_path: Path to the PDF file
            prompt_text: Prompt that includes the expected JSON structure
            model: Model name to use (will be used in the API endpoint)
            
        Returns:
            Dict containing the extracted structured information
        """
        # Process the PDF for the custom API
        pdf_data_uri = self._process_pdf_for_custom_api(pdf_path)
        
        try:
            # Prepare the request payload for the new API
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "file", "file": {"file_data": pdf_data_uri}},
                        ],
                    }
                ],
                "temperature": 0
            }
            
            # Make the API call to the new endpoint
            litellm_base_url = os.getenv("LITELLM_BASE_URL")
            if not litellm_base_url:
                error_msg = "LITELLM_BASE_URL environment variable is not set"
                logging.error(error_msg)
                raise ValueError(error_msg)
            url = f"{litellm_base_url.rstrip('/')}/openai/deployments/{model}/chat/completions"
            
            # Get API key from environment variable
            litellm_api_key = os.getenv("LITELLM_API_KEY")
            if not litellm_api_key:
                error_msg = "LITELLM_API_KEY environment variable is not set"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {litellm_api_key}"
            }
            logging.info(f"[GEMINI_OCR] Model Used: {model}")
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            # Log response details
            # logging.info(f"[GEMINI_OCR] Response: {response}")
            # logging.info(f"[GEMINI_OCR] Response status code: {response.status_code}")
            # logging.info(f"[GEMINI_OCR] Response text: {response.text}")
            if response.status_code != 200:
                logging.error(f"API request failed with status {response.status_code}: {response.text}")
                return {"error": f"API request failed: {response.status_code} - {response.text}"}
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from the response
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                logging.info(f"[GEMINI_OCR] Raw content from API: {content}")
                
                # Try to parse as JSON, but handle plain text responses
                try:
                    # Check if content is wrapped in markdown code blocks
                    if content.strip().startswith('```json'):
                        # Extract JSON from markdown code blocks
                        json_start = content.find('```json') + 7  # Skip ```json
                        json_end = content.rfind('```')
                        if json_end > json_start:
                            json_content = content[json_start:json_end].strip()
                            logging.info(f"[GEMINI_OCR] Extracted JSON from markdown: {json_content[:200]}...")
                            structured_output = json.loads(json_content)
                        else:
                            raise ValueError("Invalid markdown code block format")
                    else:
                        # Try to parse as regular JSON
                        structured_output = json.loads(content)
                except json.JSONDecodeError:
                    # If it's not JSON, treat it as plain text
                    logging.info(f"[GEMINI_OCR] Content is not JSON, treating as plain text")
                    structured_output = {"text": content}
            else:
                raise ValueError("Invalid response format from API")
            
            logging.info(f"[GEMINI_OCR] Structured output: {structured_output}")
            
            # Get token usage if available
            total_tokens = 0
            if 'usage' in response_data:
                total_tokens = response_data['usage'].get('total_tokens', 0)
            
            result_dict = {
                "structured_output": structured_output,
                "tokens_used": total_tokens,
                "processing_method": "custom_api_endpoint"
            }
            
            return result_dict
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling custom API: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"error": str(e)}

    def extract_text_with_model(
        self,
        image: Union[PIL.Image.Image, str, Path],
        reference_text: Optional[str] = None,
        prompt_text: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ) -> Dict:
        """
        Extract text from an image using the custom API endpoint with model parameter.
        
        Args:
            image: The image to process (PIL Image or file path)
            reference_text: Optional reference text to compare against
            prompt_text: Optional custom prompt to use for the LLM call
            model: Model name to use (claude-3-sonnet-20240229, claude-3-opus-20240229, etc.)
            
        Returns:
            Dict containing the extracted text and evaluation results
        """
        # Process the image for the custom API
        image_data_uri = self._process_image_for_custom_api(image)
        
        # Use the provided prompt_text if given, else fallback to the hardcoded prompt
        prompt_structure = """
**Input Provided to You:**
*   An image containing the handwritten Hindi text.
*   A string containing the reference Hindi text (this is the ground truth the student was asked to write).

**Output Format (Mandatory):**
You must produce a JSON list of objects. Each object in the list represents the evaluation of a single word from the reference text, in the order they appear. Each object must contain the following keys:

*   `reference_word` (string): The word from the reference text.
*   `transcribed_word` (string/null): The corresponding word or segment transcribed from the image.
    *   If a directly corresponding word is found, provide it.
    *   If the word seems to be part of a merged segment in the transcription (e.g., reference "मत कर" transcribed as "मतकर"), this field might show the merged segment for both reference words involved.
    *   If the word from the reference is entirely missing in the transcription, use `null` or an empty string for this field.
    *   If a word in the image is completely illegible, you can represent it as `"[illegible]"`.
*   `match` (boolean): `true` if the `transcribed_word` (or the relevant part of it) is an exact character-by-character match with the `reference_word` (including all matras and conjunct characters). `false` otherwise.
*   `reason_diff` (string):
    *   If `match` is `true`, this field can be an empty string or a brief confirmation like "Exact match."
    *   If `match` is `false`, provide a concise explanation of the mismatch. Examples include:
        *   "Spelling error: Transcribed '[transcribed]' vs reference '[reference]' (e.g., incorrect matra, different character)."
        *   "Missing matra: e.g., 'ा' missing in '[transcribed]'."
        *   "Extra character: e.g., additional 'र्' in '[transcribed]'."
        *   "Word missing: Reference word '[reference]' not found in transcription at this position."
        *   "Segmentation error: Reference '[reference]' appears merged in transcription (e.g., as part of '[merged_transcribed_segment]')."
        *   "Segmentation error: Reference '[reference]' appears split in transcription."
        *   "Illegible word in transcription."

**Detailed Instructions for Comparison and Evaluation:**
*   **Sequential Evaluation:** Iterate through the words of the reference text in order. For each `reference_word`, identify its corresponding counterpart(s) or absence in your transcribed text.
*   **Accuracy:** The comparison must be exact. Differences in matras (vowel signs), anusvara, visarga, chandrabindu, and base characters constitute a mismatch.
*   **Word Segmentation:**
    *   If the student merges words that are separate in the reference (e.g., reference "मत कर", transcribed "मतकर"), then for `reference_word: "मत"`, the `transcribed_word` could be "मतकर", `match: false`, and `reason_diff` should explain the merge. Similarly for `reference_word: "कर"`.
    *   If the student splits a word that is single in the reference, adapt the `reason_diff` accordingly.
*   **Missing/Extra Words:**
    *   If a reference word is missing from the transcription, indicate this clearly.
    *   If the transcription contains extra words not present in the reference text, these should ideally be noted after all reference words have been evaluated, perhaps as additional entries with `reference_word: null` or by detailing them in the `reason_diff` of a nearby word if they disrupt the alignment significantly. For simplicity, prioritize evaluating against the reference words first.

**Example (Conceptual):**
If Reference Text is: `हर पल`
And Transcribed Text from image is: `हर पल`
Output:
```json
[
  {
    "reference_word": "हर",
    "transcribed_word": "हर",
    "match": true,
    "reason_diff": "Exact match."
  },
  {
    "reference_word": "पल",
    "transcribed_word": "पल",
    "match": true,
    "reason_diff": "Exact match."
  }
]
```

If Reference Text is: लड़ाई
And Transcribed Text from image is: लड़ई
Output:
```json
[
  {
    "reference_word": "लड़ाई",
    "transcribed_word": "लड़ई",
    "match": false,
    "reason_diff": "Spelling error: Transcribed 'लड़ई' is missing the 'ा' (aa) matra found in 'लड़ाई'."
  }
]
```

If Reference Text is: उस तट पर
And Transcribed Text from image is: उस पर (student missed "तट")
Output:
```json
[
  {
    "reference_word": "उस",
    "transcribed_word": "उस",
    "match": true,
    "reason_diff": "Exact match."
  },
  {
    "reference_word": "तट",
    "transcribed_word": null,
    "match": false,
    "reason_diff": "Word missing: Reference word 'तट' not found in transcription at this position."
  },
  {
    "reference_word": "पर",
    "transcribed_word": "पर",
    "match": true,
    "reason_diff": "Exact match."
  }
]
```

Begin by transcribing the provided image, then proceed to the word-by-word evaluation against the reference text, structuring your final output strictly in the JSON format specified.
"""

        sample_prompt_text = """
You are an AI assistant specialized in Optical Character Recognition (OCR) and text comparison for handwritten Hindi. You will be provided with an image containing handwritten Hindi text and a corresponding reference Hindi text that the handwriting is supposed to match.

Your task is to:
1.  **Transcribe:** Accurately transcribe the Hindi words from the provided image. Focus exclusively on Hindi script and words. Ignore any non-Hindi elements.
2.  **Tokenize:** Internally, split both the reference text and your transcribed text into individual words. Word boundaries are typically defined by spaces.
3.  **Compare and Evaluate:** Perform a word-by-word comparison of your transcribed text against the reference text. Your output should be a detailed evaluation for each word based on the sequence in the reference text.
"""
        
        if prompt_text:
            prompt = prompt_text
        else:
            prompt = f"{sample_prompt_text}\n{prompt_structure}"
        
        if reference_text:
            prompt = f"{prompt}\n\nReference Text: {reference_text}"
        
        try:
            # Get API endpoint details
            litellm_base_url = os.getenv("LITELLM_BASE_URL")
            if not litellm_base_url:
                error_msg = "LITELLM_BASE_URL environment variable is not set"
                logging.error(error_msg)
                raise ValueError(error_msg)
            url = f"{litellm_base_url.rstrip('/')}/openai/deployments/{model}/chat/completions"

            # Get API key from environment variable
            litellm_api_key = os.getenv("LITELLM_API_KEY")
            if not litellm_api_key:
                error_msg = "LITELLM_API_KEY environment variable is not set"
                logging.error(error_msg)
                raise ValueError(error_msg)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {litellm_api_key}"
            }

            # Retry logic with temperature variation
            max_retries = 3
            temperatures = [0, 0.3, 0.5]
            retry_count = 0
            evaluations = None
            response_data = None
            content = None

            while retry_count < max_retries:
                temperature = temperatures[retry_count]
                logging.info(f"[GEMINI_OCR] Attempt {retry_count + 1}/{max_retries} with temperature={temperature}")

                try:
                    # Prepare the request payload for the custom API
                    payload = {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": image_data_uri},
                                ],
                            }
                        ],
                        "temperature": temperature
                    }

                    # Make the API call
                    response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

                    # Log response details
                    logging.info(f"[GEMINI_OCR] Response: {response}")
                    logging.info(f"[GEMINI_OCR] Response status code: {response.status_code}")
                    logging.info(f"[GEMINI_OCR] Response text: {response.text}")

                    if response.status_code != 200:
                        logging.error(f"API request failed with status {response.status_code}: {response.text}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            return {"error": f"API request failed: {response.status_code} - {response.text}"}
                        continue

                except requests.exceptions.Timeout as e:
                    logging.warning(f"[GEMINI_OCR] Timeout on attempt {retry_count + 1}/{max_retries}: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"[GEMINI_OCR] Max retries reached after timeout")
                        return {"error": f"API request timed out after {max_retries} attempts"}
                    logging.info(f"[GEMINI_OCR] Retrying after timeout...")
                    continue

                except requests.exceptions.RequestException as e:
                    logging.warning(f"[GEMINI_OCR] Request error on attempt {retry_count + 1}/{max_retries}: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"[GEMINI_OCR] Max retries reached after request error")
                        return {"error": f"API request failed: {str(e)}"}
                    logging.info(f"[GEMINI_OCR] Retrying after request error...")
                    continue

                # Parse the response
                response_data = response.json()

                # Extract the content from the response
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    content = response_data['choices'][0]['message']['content']
                    logging.info(f"[GEMINI_OCR] Raw content from API: {content[:200]}...")

                    # Try to extract and parse JSON
                    try:
                        json_content = None

                        # Check if content is wrapped in markdown code blocks
                        if '```json' in content:
                            # Extract JSON from markdown code blocks
                            json_start = content.find('```json') + 7  # Skip ```json
                            json_end = content.rfind('```')
                            if json_end > json_start:
                                json_content = content[json_start:json_end].strip()
                                logging.info(f"[GEMINI_OCR] Extracted JSON from markdown code blocks")
                            else:
                                raise ValueError("Invalid markdown code block format")
                        elif content.strip().startswith('['):
                            # Content is already clean JSON
                            json_content = content.strip()
                            logging.info(f"[GEMINI_OCR] Content is already clean JSON")
                        else:
                            # Try to find JSON array anywhere in the content
                            start_idx = content.find('[')
                            if start_idx != -1:
                                # Find the matching closing bracket
                                bracket_count = 0
                                for i in range(start_idx, len(content)):
                                    if content[i] == '[':
                                        bracket_count += 1
                                    elif content[i] == ']':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            json_content = content[start_idx:i+1]
                                            logging.info(f"[GEMINI_OCR] Extracted JSON array from text")
                                            break

                            if not json_content:
                                raise ValueError("Could not find JSON array in content")

                        # Try to parse the extracted JSON
                        evaluations = json.loads(json_content)

                        # Validate structure
                        if not isinstance(evaluations, list) or len(evaluations) == 0:
                            raise ValueError("Parsed JSON is not a non-empty array")

                        # Check first item has required keys
                        required_keys = ['reference_word', 'transcribed_word', 'match']
                        if not all(key in evaluations[0] for key in required_keys):
                            raise ValueError(f"Missing required keys in evaluation object")

                        # Successfully parsed and validated!
                        logging.info(f"[GEMINI_OCR] Successfully parsed JSON with {len(evaluations)} evaluations")
                        break  # Exit retry loop

                    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                        logging.warning(f"[GEMINI_OCR] Parsing error on attempt {retry_count + 1}: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logging.error(f"[GEMINI_OCR] Max retries reached after parsing errors")
                            return {"error": f"Failed to parse JSON response: {str(e)}", "raw_output": content[:500]}
                        logging.info(f"[GEMINI_OCR] Retrying... (attempt {retry_count + 1}/{max_retries})")
                        continue
                else:
                    raise ValueError("Invalid response format from API")

            logging.info(f"[GEMINI_OCR] Parsed evaluations: {evaluations}")
            
            # Validate evaluations against our schema
            evaluations = transform_word_evaluations_nl(evaluations)
            word_evaluations = [WordEvaluation(**eval_data) for eval_data in evaluations]
            
            # Construct the full text from transcribed words
            transcribed_words = []
            for eval in word_evaluations:
                if eval.transcribed_word and eval.transcribed_word != "[illegible]":
                    transcribed_words.append(eval.transcribed_word)
            
            full_text = " ".join(transcribed_words)
            
            # Calculate accuracy metrics
            total_words = len(word_evaluations)
            correct_words = sum(1 for eval in word_evaluations if eval.match)
            accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
            
            # Get token usage if available
            total_tokens = 0
            if 'usage' in response_data:
                total_tokens = response_data['usage'].get('total_tokens', 0)
            
            result_dict = {
                "full_text": full_text,
                "evaluations": [eval.dict() for eval in word_evaluations],
                "accuracy": accuracy,
                "correct_words": correct_words,
                "total_words": total_words,
                "tokens_used": total_tokens,
                "processing_method": "custom_api_endpoint"
            }

            return result_dict

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"error": str(e)}

    def compare_json_structures(
        self,
        expected_json: Dict,
        llm_output_json: Dict,
        comparison_prompt: Optional[str] = None
    ) -> Dict:
        """
        Compare two JSON structures and evaluate semantic similarity.
        
        Args:
            expected_json: The expected JSON structure
            llm_output_json: The JSON output from LLM
            comparison_prompt: Optional custom prompt for comparison
            
        Returns:
            Dict containing comparison results and similarity scores
        """
        # Prepare the comparison prompt
        if not comparison_prompt:
            comparison_prompt = """
You are an AI assistant specialized in comparing JSON structures and evaluating semantic similarity.

Compare the two JSON structures provided and evaluate how semantically similar the values are under each key. 
Consider the meaning and intent rather than exact string matches.

For each key present in either JSON:
1. Evaluate semantic similarity (0-1 scale)
2. Provide a brief explanation of the comparison
3. Note any missing or extra keys
4. For the fields 'expected_value' and 'actual_value', always include the FULL value from the JSON, even if it is a long array or object. Do NOT summarize, truncate, or use ellipses. Output the entire array or object as it appears in the input.**

Return your analysis in the following JSON format:
{
    "overall_similarity": 0.85,
    "key_comparisons": {
        "key_name": {
            "similarity": 0.9,
            "explanation": "Both values represent the same concept with minor differences",
            "expected_value": "<full value from expected_json>",
            "actual_value": "<full value from llm_output_json>"
        }
    },
    "missing_keys": ["keys present in expected but not in actual"],
    "extra_keys": ["keys present in actual but not in expected"]
}
"""
        
        # Prepare the data for comparison
        comparison_data = {
            "expected_json": expected_json,
            "llm_output_json": llm_output_json
        }
        
        try:
            # Prepare configuration
            config = {
                "temperature": 0,
                "response_mime_type": "application/json",
            }
            
            # Create the prompt with both JSONs
            full_prompt = f"{comparison_prompt}\n\nExpected JSON:\n{json.dumps(expected_json, indent=2)}\n\nLLM Output JSON:\n{json.dumps(llm_output_json, indent=2)}"
            
            # Log the prompt for debugging
            logging.info(f"[GeminiOCR] Comparison prompt (first 1000 chars): {full_prompt[:1000]}")
            logging.info(f"[GeminiOCR] Comparison prompt (last 1000 chars): {full_prompt[-1000:]}")
            
            # Make the API call
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[full_prompt],
                config=config,
            )
            
            # Extract and parse the response
            output = response.candidates[0].content.parts[0].text
            logging.info(f"[GeminiOCR] Gemini comparison raw output (first 1000 chars): {output[:1000]}")
            logging.info(f"[GeminiOCR] Gemini comparison raw output (last 1000 chars): {output[-1000:]}")
            comparison_results = json.loads(output)
            
            # Get token usage
            total_tokens = 0
            if response.usage_metadata:
                total_tokens = response.usage_metadata.total_token_count
            
            comparison_results["tokens_used"] = total_tokens
            
            return comparison_results
            
        except APIError as e:
            logging.error(f"Error calling Gemini API for comparison: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse comparison JSON response: {e}")
            logging.error(f"Raw output that failed to parse (first 2000 chars): {output[:2000]}")
            logging.error(f"Raw output that failed to parse (last 2000 chars): {output[-2000:]}")
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected error during comparison: {e}")
            return {"error": str(e)}
    
    def compare_json_structures_keywise(
        self,
        expected_json: Dict,
        llm_output_json: Dict,
        comparison_prompt: Optional[str] = None
    ) -> Dict:
        """
        Compare two JSON structures key-wise, running LLM comparisons for each key in parallel.
        Args:
            expected_json: The expected JSON structure
            llm_output_json: The JSON output from LLM
            comparison_prompt: Optional custom prompt for comparison
        Returns:
            Dict containing overall similarity, per-key results, missing/extra keys, and token usage
        """
        import concurrent.futures

        # 1. Collect all unique keys
        expected_keys = set(expected_json.keys())
        output_keys = set(llm_output_json.keys())
        all_keys = expected_keys | output_keys
        missing_keys = list(expected_keys - output_keys)
        extra_keys = list(output_keys - expected_keys)

        # 2. Prepare default prompt if not provided
        if not comparison_prompt:
            comparison_prompt = (
                "You are an AI assistant specialized in comparing JSON values and evaluating semantic similarity.\n"
                "Compare the two values provided for the key below and evaluate how semantically similar they are. "
                "Consider the meaning and intent rather than exact string matches.\n"
                "Return your analysis in the following JSON format:\n"
                "{\n"
                "  'similarity': 0.9,\n"
                "  'explanation': 'Both values represent the same concept with minor differences',\n"
                "  'expected_value': <full value from expected_json>,\n"
                "  'actual_value': <full value from llm_output_json>\n"
                "}"
            )

        # 3. Function to compare a single key
        def compare_key(key):
            expected_value = expected_json.get(key, None)
            actual_value = llm_output_json.get(key, None)
            prompt = (
                f"{comparison_prompt}\n\n"
                f"Key: {key}\n"
                f"Expected Value:\n{json.dumps(expected_value, ensure_ascii=False, indent=2)}\n\n"
                f"Actual Value:\n{json.dumps(actual_value, ensure_ascii=False, indent=2)}"
            )
            config = {
                "temperature": 0,
                "response_mime_type": "application/json",
            }
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[prompt],
                    config=config,
                )
                output = response.candidates[0].content.parts[0].text
                result = json.loads(output)
                tokens_used = 0
                if response.usage_metadata:
                    tokens_used = response.usage_metadata.total_token_count
                result["tokens_used"] = tokens_used
                return key, result
            except Exception as e:
                return key, {"error": str(e), "expected_value": expected_value, "actual_value": actual_value}

        # 4. Run all key comparisons in parallel
        key_comparisons = {}
        total_tokens = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {executor.submit(compare_key, key): key for key in all_keys}
            for future in concurrent.futures.as_completed(future_to_key):
                key, result = future.result()
                key_comparisons[key] = result
                if isinstance(result, dict) and "tokens_used" in result:
                    total_tokens += result["tokens_used"]

        # 5. Compute overall similarity (mean of available similarities)
        similarities = [v["similarity"] for v in key_comparisons.values() if isinstance(v, dict) and "similarity" in v]
        overall_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return {
            "overall_similarity": overall_similarity,
            "key_comparisons": key_comparisons,
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
            "tokens_used": total_tokens
        }
        
   
     
def main():
    """Example usage of the GeminiOCR class."""
    # Example usage
    ocr = GeminiOCR()
    
    # Example image path (replace with your image path)
    image_path = "path/to/your/image.jpg"
    reference_text = "हर पल लड़ाई मत कर"
    
    try:
        result = ocr.extract_text(image_path, reference_text)
        print("Extracted Text and Evaluation:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 