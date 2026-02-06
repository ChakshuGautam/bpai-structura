import os
import sys
# Add marker directory to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../marker"))

from dotenv import load_dotenv
load_dotenv()

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import modal
import os
import json
import uuid


from core.pdf_domain_converter import PdfConverterCustom
from core.execution.s3 import download_from_s3, save_json_to_s3, upload_to_s3
from core.execution.modal_image import marker_image


# Use the cached marker_image from modal_image.py
print("Using cached Modal image for Marker...")

# TODO: Deploy rather than run
# App name is configurable via MODAL_APP_NAME env var (default: "marker")
# This allows deploying multiple versions side-by-side for A/B testing
app_name = os.environ.get("MODAL_APP_NAME", "marker")
print(f"Creating Modal app: {app_name}...")
# Create a Modal app
app = modal.App(app_name)
config = {
    "output_format": "json",
    "use_llm": "true",
    "gemini_api_key": None,  # Will be set from GOOGLE_API_KEY env var inside function
    "use_gemini_correction": True,  # Enable Gemini HTML correction post-processing
    "paginate_output": False,  # Disable pagination to avoid schema errors
    "max_remote_calls": 50  # Increase parallel Gemini API calls (default: 10)
}

# TODO: Move these to part of the container (takes time)
# Note: Converter is initialized inside the Modal function, not at module level
# to avoid local initialization during deployment

@app.function(image=marker_image, gpu="A100", timeout=1800)
def parse_pdf(url: str):
    """Parse a PDF and save results to S3 bucket"""
    # Marker is already installed in the container
    import redis
    from dotenv import load_dotenv

    # Load .env from Modal container path
    load_dotenv("/root/.env")

    # Update config with API key from environment
    config["gemini_api_key"] = os.environ.get("GOOGLE_API_KEY")
    print(f"Using GOOGLE_API_KEY from .env: {config['gemini_api_key'][:20]}...")

    # Initialize the converter inside the function
    config_parser = ConfigParser(config)
    converter = PdfConverterCustom(
        config=config_parser.generate_config_dict(),
        renderer=config_parser.get_renderer(),
        artifact_dict=create_model_dict(),
    )

    # Extract task_id from the URL (assuming URL format ends with task_id)
    # This is needed if we're using the server's task_id for S3 storage
    parts = url.split("/")
    task_id = parts[-1]

    path = download_from_s3(url)
    rendered = converter(path)
    text, _, _ = text_from_rendered(rendered)

    # Check if the text is already a JSON string and convert to a Python object if needed
    # This ensures we don't double-encode when saving to S3
    if isinstance(text, str):
        try:
            # Try to parse it as JSON
            parsed_text = json.loads(text)
            text = parsed_text
        except json.JSONDecodeError:
            # If it's not valid JSON, leave it as is
            pass

    # Post-process with Gemini HTML correction if enabled
    if config.get("use_gemini_correction", False):
        from core.processors.gemini_html_corrector import correct_html_with_gemini
        try:
            print("Applying Gemini HTML correction...")
            text = correct_html_with_gemini(
                pdf_path=path,
                json_data=text,
                api_key=os.environ.get("GOOGLE_API_KEY"),
                dpi=100,
                max_workers=5
            )
            print("Gemini HTML correction complete")
        except Exception as e:
            print(f"Gemini HTML correction failed: {e}")
            # Continue without correction if it fails
    
    # Save results to S3 using the function from s3.py
    bucket_name = "pdf-results"
    print(f"Saving results for task_id: {task_id}")
    result_url, result_id = save_json_to_s3(text, bucket_name, task_id=task_id)
    print(f"Saved to S3 - URL: {result_url}, ID: {result_id}")

    # Push notification to Redis
    redis_kwargs = {
        "host": os.environ.get("REDIS_HOST", "localhost"),
        "port": int(os.environ.get("REDIS_PORT", 6379))
    }
    
    # Add authentication if credentials are provided
    redis_username = os.environ.get("REDIS_USERNAME")
    redis_password = os.environ.get("REDIS_PASSWORD")
    
    if redis_username and redis_password:
        redis_kwargs.update({
            "username": redis_username,
            "password": redis_password
        })
    elif redis_password:
        redis_kwargs["password"] = redis_password
    
    redis_client = redis.Redis(**redis_kwargs)
    
    notification = {
        "status": "completed",
        "result_id": result_id,
        "result_url": result_url,
        "source_url": url,
        "task_id": task_id  # Include task_id in notification
    }
    
    # Publish completion notification
    redis_channel = os.environ.get("REDIS_CHANNEL", "pdf_processing")
    print(f"Publishing to Redis channel: {redis_channel}")
    print(f"Notification: {json.dumps(notification, indent=2)}")
    redis_client.publish(redis_channel, json.dumps(notification))
    print("Redis notification published successfully")

    return {
        "status": "completed",
        "result_url": result_url,
        "task_id": task_id
    }
    
@app.local_entrypoint()
def main():
    # filepath = "/Users/__chaks__/Desktop/BPAI-Dubious.pdf"
    # filepath = "/Users/__chaks__/Desktop/BPAI-Table.pdf"
    # filepath = "/Users/__chaks__/Desktop/BPAI.pdf"
    filepath = "/C:/Users/debat/OneDrive/Desktop/BPAI.pdf"
    filepath = "/Users/__chaks__/Desktop/Purchase Contract.pdf"
    # filepath = "/Users/__chaks__/Desktop/BPAI-p1.pdf"
    url = upload_to_s3(filepath, "marker-pdf")
    print(f"Uploaded to S3: {url}")
    # url = "https://structura.blr1.digitaloceanspaces.com/marker-pdf/abe6ffe9-acbf-441b-83c3-724130ec9e27"
    result_url = parse_pdf.remote(url)
    print(f"Processing complete. Results saved to: {result_url}")