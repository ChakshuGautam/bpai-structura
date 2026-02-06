"""
Databricks-side PDF processing script.

Runs ON Databricks as a Python task within a GPU-enabled cluster.
Mirrors the pipeline from core/execution/modal_execute.py:parse_pdf.

Usage: Triggered via Databricks Jobs API with parameter:
  s3_url - The S3 URL of the PDF to process (with task_id appended)
"""
import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_secrets():
    """Load secrets from Databricks secret scope 'structura'."""
    # dbutils is available in the Databricks runtime environment
    from pyspark.dbutils import DBUtils
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)

    secrets = {
        "GOOGLE_API_KEY": dbutils.secrets.get("structura", "GOOGLE_API_KEY"),
        "DO_ENDPOINT_URL": dbutils.secrets.get("structura", "DO_ENDPOINT_URL"),
        "DO_ACCESS_KEY_ID": dbutils.secrets.get("structura", "DO_ACCESS_KEY_ID"),
        "DO_SECRET_ACCESS_KEY": dbutils.secrets.get("structura", "DO_SECRET_ACCESS_KEY"),
        "REDIS_HOST": dbutils.secrets.get("structura", "REDIS_HOST"),
        "REDIS_PORT": dbutils.secrets.get("structura", "REDIS_PORT"),
        "REDIS_USERNAME": dbutils.secrets.get("structura", "REDIS_USERNAME"),
        "REDIS_PASSWORD": dbutils.secrets.get("structura", "REDIS_PASSWORD"),
    }

    # Set as environment variables so downstream modules (s3.py, etc.) pick them up
    for key, value in secrets.items():
        if value:
            os.environ[key] = value

    return secrets


def get_job_parameter(name: str) -> str:
    """Read a job parameter passed via Databricks widgets."""
    from pyspark.dbutils import DBUtils
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)

    dbutils.widgets.text(name, "")
    return dbutils.widgets.get(name)


def parse_pdf(url: str) -> dict:
    """
    Parse a PDF and save results to S3.

    Mirrors core/execution/modal_execute.py:parse_pdf exactly:
    1. Download PDF from S3
    2. Initialize Marker converter with GPU models
    3. Convert PDF -> JSON
    4. Apply Gemini HTML correction
    5. Save results to S3 (pdf-results/{task_id}.json)
    6. Publish Redis notification
    """
    import redis
    import boto3
    import uuid
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered
    from core.pdf_domain_converter import PdfConverterCustom
    from core.execution.s3 import download_from_s3, save_json_to_s3

    # Extract task_id from URL (format: s3_url/task_id)
    parts = url.split("/")
    task_id = parts[-1]

    logger.info(f"Processing task_id: {task_id}")
    logger.info(f"Source URL: {url}")

    # Configure the Marker converter
    config = {
        "output_format": "json",
        "use_llm": "true",
        "gemini_api_key": os.environ.get("GOOGLE_API_KEY"),
        "use_gemini_correction": True,
        "paginate_output": False,
        "max_remote_calls": 50,
    }

    logger.info("Initializing Marker converter with GPU models...")
    config_parser = ConfigParser(config)
    converter = PdfConverterCustom(
        config=config_parser.generate_config_dict(),
        renderer=config_parser.get_renderer(),
        artifact_dict=create_model_dict(),
    )

    # Download PDF from S3
    logger.info("Downloading PDF from S3...")
    path = download_from_s3(url)

    # Convert PDF -> JSON
    logger.info("Converting PDF...")
    rendered = converter(path)
    text, _, _ = text_from_rendered(rendered)

    # Ensure text is a Python object, not double-encoded JSON
    if isinstance(text, str):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            pass

    # Post-process with Gemini HTML correction
    if config.get("use_gemini_correction", False):
        from core.processors.gemini_html_corrector import correct_html_with_gemini
        try:
            logger.info("Applying Gemini HTML correction...")
            text = correct_html_with_gemini(
                pdf_path=path,
                json_data=text,
                api_key=os.environ.get("GOOGLE_API_KEY"),
                dpi=100,
                max_workers=5,
            )
            logger.info("Gemini HTML correction complete")
        except Exception as e:
            logger.error(f"Gemini HTML correction failed: {e}")

    # Save results to S3
    bucket_name = "pdf-results"
    logger.info(f"Saving results for task_id: {task_id}")
    result_url, result_id = save_json_to_s3(text, bucket_name, task_id=task_id)
    logger.info(f"Saved to S3 - URL: {result_url}, ID: {result_id}")

    # Publish Redis notification
    redis_kwargs = {
        "host": os.environ.get("REDIS_HOST", "localhost"),
        "port": int(os.environ.get("REDIS_PORT", 6379)),
    }

    redis_username = os.environ.get("REDIS_USERNAME")
    redis_password = os.environ.get("REDIS_PASSWORD")

    if redis_username and redis_password:
        redis_kwargs.update({
            "username": redis_username,
            "password": redis_password,
        })
    elif redis_password:
        redis_kwargs["password"] = redis_password

    redis_client = redis.Redis(**redis_kwargs)

    notification = {
        "status": "completed",
        "result_id": result_id,
        "result_url": result_url,
        "source_url": url,
        "task_id": task_id,
    }

    redis_channel = os.environ.get("REDIS_CHANNEL", "pdf_processing")
    logger.info(f"Publishing to Redis channel: {redis_channel}")
    logger.info(f"Notification: {json.dumps(notification, indent=2)}")
    redis_client.publish(redis_channel, json.dumps(notification))
    logger.info("Redis notification published successfully")

    return {
        "status": "completed",
        "result_url": result_url,
        "task_id": task_id,
    }


if __name__ == "__main__":
    # Entry point when run as a Databricks Python task
    logger.info("Starting Structura parse_pdf on Databricks...")

    # Load secrets from Databricks secret scope
    get_secrets()

    # Read job parameter
    s3_url = get_job_parameter("s3_url")
    if not s3_url:
        raise ValueError("Missing required job parameter: s3_url")

    logger.info(f"Received s3_url: {s3_url}")

    result = parse_pdf(s3_url)
    logger.info(f"Processing complete: {json.dumps(result, indent=2)}")
