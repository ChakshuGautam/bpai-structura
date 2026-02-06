"""
Converts PDF to JSON using Marker API
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MarkerAPICall:
    """Class to handle Marker API calls and processing"""
    
    def __init__(self, pdf_path: str, config: Optional[Dict[str, Any]] = None):
        """Initialize MarkerAPICall with PDF path and optional config.
        
        Args:
            pdf_path: Path to the PDF file
            config: Optional configuration dictionary
        """
        self.pdf_path = pdf_path
        self.config = config or {}
        self.marker_output = None
        
        # Get API key from environment or config
        self.api_key = self.config.get('marker_api_key') or os.environ.get("MARKER_API_KEY")
        if not self.api_key:
            raise ValueError("MARKER_API_KEY environment variable or config not set")
            
        # API endpoints
        self.base_url = "https://www.datalab.to/api/v1/marker"
        
        # Configure request headers with X-API-Key
        self.headers = {
            'accept': 'application/json',
            'X-API-Key': self.api_key
        }

    def get_marker_data(self, file_path: str) -> Dict[str, Any]:
        """Process PDF through Marker API and get results.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict containing the Marker API response JSON
        """
        try:
            # Prepare file upload
            files = {
                'file': (os.path.basename(file_path), 
                        open(file_path, 'rb'), 
                        'application/pdf')
            }
            
            # Configure form data
            form_data = {
                'output_format': 'json',
                'include_text': 'true',
                'include_structure': 'true',
                'include_layout': 'true',
                'include_character_breaks': 'true',
                'extract_tables': 'true',
                'table_segmentation_mode': 'cells',
                'include_formulas': 'true', 
                'include_meta': 'true',
                'detect_lists': 'true',
                'language': 'auto',
                'force_ocr': 'true',
                'process_background': 'false',
                'use_llm': 'true'
            }
            
            # Submit PDF for processing with X-API-Key header
            logger.info(f"Submitting PDF to Marker API: {file_path}")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=form_data,
                files=files,
                timeout=300
            )
            
            # Log response for debugging
            logger.debug(f"API Response Status: {response.status_code}")
            logger.debug(f"API Response Headers: {response.headers}")
            
            if response.status_code != 200:
                error_msg = f"Marker API request failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)
                
            result = response.json()
            
            if not result.get('success'):
                raise Exception(f"Marker API processing failed: {result.get('error')}")
                
            request_id = result.get('request_id')
            request_check_url = result.get('request_check_url')
            
            # Poll for results with same X-API-Key header
            max_attempts = 30
            attempt = 0
            poll_interval = 10  # seconds
            
            while attempt < max_attempts:
                attempt += 1
                logger.info(f"Checking API status, attempt {attempt}/{max_attempts}")
                
                check_response = requests.get(
                    request_check_url,
                    headers=self.headers,
                    timeout=60
                )
                
                if check_response.status_code != 200:
                    logger.warning(f"API status check failed: {check_response.status_code}")
                    time.sleep(poll_interval)
                    continue
                
                check_result = check_response.json()
                status = check_result.get('status')
                
                if status in ('completed', 'complete'):
                    logger.info("Marker API processing completed")
                    self.marker_output = check_result.get('json', {})
                    return self.marker_output
                    
                elif status == 'failed':
                    raise Exception(f"Marker API processing failed: {check_result.get('error')}")
                    
                else:
                    logger.info(f"Status: {status}, waiting {poll_interval} seconds...")
                    time.sleep(poll_interval)
                    poll_interval = min(30, poll_interval * 1.5)
            
            raise Exception("Timed out waiting for Marker API to complete")
            
        except Exception as e:
            logger.error(f"Error in Marker API processing: {str(e)}")
            raise
            
        finally:
            # Clean up file handle if needed
            if 'files' in locals() and files.get('file'):
                try:
                    files['file'][1].close()
                except:
                    pass
    
    def convert_pdf(self) -> Dict[str, Any]:
        """Convert PDF to JSON using Marker API.
        
        Returns:
            Dict containing the processed document data
        """
        if self.marker_output is None:
            self.marker_output = self.get_marker_data(self.pdf_path)
        return self.marker_output
        
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Dict containing configuration
        """
        return self.config
        
    def get_converter(self) -> Any:
        """Get API converter info.
        
        Returns:
            API info dictionary
        """
        return {
            'type': 'marker_api',
            'base_url': self.base_url
        }
        
    def get_pdf_path(self) -> str:
        """Get PDF file path.
        
        Returns:
            String path to PDF file
        """
        return self.pdf_path
    
    def get_marker_output(self) -> Dict[str, Any]:
        """Get marker output, converting if not already done.
        
        Returns:
            Dict containing processed document data
        """
        if self.marker_output is None:
            self.convert_pdf()
        return self.marker_output

    def submit_pdf(self, file_path: str) -> dict:
        """Submit PDF to Marker API and return request_id and request_check_url (no polling)."""
        files = {
            'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')
        }
        form_data = {
            'output_format': 'json',
            'include_text': 'true',
            'include_structure': 'true',
            'include_layout': 'true',
            'include_character_breaks': 'true',
            'extract_tables': 'true',
            'table_segmentation_mode': 'cells',
            'include_formulas': 'true',
            'include_meta': 'true',
            'detect_lists': 'true',
            'language': 'auto',
            'force_ocr': 'true',
            'process_background': 'false'
        }
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=form_data,
                files=files,
                timeout=300
            )
            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    error_msg = f"Marker API request failed: {error_detail}"
                except:
                    error_msg = f"Marker API request failed: {response.text}"
                raise Exception(error_msg)
            result = response.json()
            if not result.get('success'):
                raise Exception(f"Marker API processing failed: {result.get('error')}")
            self.request_id = result.get('request_id')
            self.request_check_url = result.get('request_check_url')
            return {
                'request_id': self.request_id,
                'request_check_url': self.request_check_url
            }
        finally:
            if 'files' in locals() and files.get('file'):
                try:
                    files['file'][1].close()
                except:
                    pass

    def poll_result(self, request_check_url: str) -> dict:
        """Poll Marker API for result using request_check_url."""
        response = requests.get(request_check_url, headers=self.headers, timeout=60)
        if response.status_code != 200:
            try:
                error_detail = response.json()
                error_msg = f"Marker API poll failed: {error_detail}"
            except:
                error_msg = f"Marker API poll failed: {response.text}"
            raise Exception(error_msg)
        return response.json()