import boto3
import os
from dotenv import load_dotenv
import uuid
import json
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DO_ENDPOINT_URL = os.environ.get("DO_ENDPOINT_URL")
DO_SECRET_ACCESS_KEY = os.environ.get("DO_SECRET_ACCESS_KEY")
DO_ACCESS_KEY_ID = os.environ.get("DO_ACCESS_KEY_ID")

# Check if environment variables are set
if not all([DO_ENDPOINT_URL, DO_SECRET_ACCESS_KEY, DO_ACCESS_KEY_ID]):
    raise ValueError("Required environment variables are not set")


def upload_to_s3(file_path: str, bucket_name: str):
    """Upload a file to S3 compatible storage and return the URL"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    session = boto3.session.Session()
    client = session.client('s3',
                           endpoint_url=DO_ENDPOINT_URL,
                           aws_access_key_id=DO_ACCESS_KEY_ID,
                           aws_secret_access_key=DO_SECRET_ACCESS_KEY)
    new_uuid = str(uuid.uuid4())
    
    # Use the provided bucket name instead of hardcoded "dev"
    client.upload_file(file_path, bucket_name, new_uuid)
    
    # Construct a proper URL - format depends on the S3 provider
    if DO_ENDPOINT_URL.startswith("http"):
        base_url = DO_ENDPOINT_URL
    else:
        base_url = f"https://{DO_ENDPOINT_URL}"
        
    return f"{base_url}/{bucket_name}/{new_uuid}"


def download_from_s3(url: str):
    """Download a file from S3 compatible storage and return the local path"""
    # Check if the URL has a task_id appended (format: s3_url/task_id)
    # If so, strip it out for S3 operations
    if url.count('/') > 4:  # Basic check for extra path segments
        # Split by forward slash and look for a UUID-like pattern at the end
        parts = url.split('/')
        last_part = parts[-1]
        if len(last_part) >= 32 and '-' in last_part:  # Simple UUID check
            # This looks like a task_id, so reconstruct the URL without it
            url = '/'.join(parts[:-1])
    
    session = boto3.session.Session()
    client = session.client('s3',
                           endpoint_url=DO_ENDPOINT_URL,
                           aws_access_key_id=DO_ACCESS_KEY_ID,
                           aws_secret_access_key=DO_SECRET_ACCESS_KEY)
    
    # Parse the URL to extract bucket and key
    # This parsing logic might need adjustment based on your actual URL format
    parts = url.split("/")
    if len(parts) < 5:
        raise ValueError(f"Invalid S3 URL format: {url}")
    
    bucket_name = parts[-2]
    key = parts[-1]
    
    temp_path = f"/tmp/{key}"
    
    client.download_file(bucket_name, key, temp_path)
    return temp_path


def save_json_to_s3(data: dict, bucket_name: str, task_id: str = None, prefix: str = ""):
    """Save JSON data to S3 compatible storage and return the URL
    
    Args:
        data: The data to save as JSON
        bucket_name: S3 bucket name
        task_id: Task ID to use as the filename (if not provided, a new UUID will be generated)
        prefix: Optional prefix for the key/filename (e.g. "results/")
        
    Returns:
        tuple: (result_url, result_id)
    """
    logger.info(f"Saving data to S3. Type: {type(data)}")
    if isinstance(data, str):
        logger.info(f"Data appears to be a string. First 100 chars: {data[:100]}")
    
    session = boto3.session.Session()
    client = session.client('s3',
                           endpoint_url=DO_ENDPOINT_URL,
                           aws_access_key_id=DO_ACCESS_KEY_ID,
                           aws_secret_access_key=DO_SECRET_ACCESS_KEY)
    
    # Use provided task_id or generate a unique ID
    result_id = task_id if task_id else str(uuid.uuid4())
    
    # Create the key with optional prefix
    key = f"{prefix}{result_id}.json" if prefix else f"{result_id}.json"
    
    # Make sure data is a proper Python object, not a JSON string
    if isinstance(data, str):
        try:
            # If data is already a JSON string, parse it to ensure we don't double-encode
            data = json.loads(data)
            logger.info("Successfully parsed string data as JSON")
        except json.JSONDecodeError as e:
            # If it's just a regular string, that's fine, it will be serialized properly
            logger.warning(f"Data is a string but not valid JSON: {str(e)}")
    
    # Upload the JSON data
    serialized_data = json.dumps(data)
    logger.info(f"Serialized data length: {len(serialized_data)}")
    logger.info(f"First 100 chars of serialized data: {serialized_data[:100]}")
    
    client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=serialized_data,
        ContentType="application/json"
    )
    
    # Construct the result URL
    if DO_ENDPOINT_URL.startswith("http"):
        base_url = DO_ENDPOINT_URL
    else:
        base_url = f"https://{DO_ENDPOINT_URL}"
    
    result_url = f"{base_url}/{bucket_name}/{key}"

    logger.info(f"Successfully saved to S3:")
    logger.info(f"  Bucket: {bucket_name}")
    logger.info(f"  Key: {key}")
    logger.info(f"  Result URL: {result_url}")
    logger.info(f"  Result ID: {result_id}")

    return result_url, result_id 