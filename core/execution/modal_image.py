import modal
import os

# Get the absolute path to the marker directory
marker_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../marker"))

# Define the Modal image to be built on Modal's infrastructure
marker_image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install system dependencies (poppler-utils needed for pdf2image)
    .apt_install("git", "poppler-utils")
    # Install Python packages
    .pip_install("boto3", "redis", "poetry==1.8.2", "pdf2image", "google-genai", "pillow")
    # Add env file
    .add_local_file(".env", "/root/.env", copy=True)
    # Add the marker source code
    .add_local_dir(marker_source_dir, "/root/marker", copy=True)
    # Run commands to install marker using Poetry
    .run_commands(
        "cd /root/marker && "
        "poetry config virtualenvs.create false && "
        "poetry install --no-interaction --no-ansi && "
        "python -c 'from marker.models import create_model_dict; create_model_dict()'"
    )
)

marker_image = marker_image.add_local_python_source(
    "core"
)

__all__ = ["marker_image"] 