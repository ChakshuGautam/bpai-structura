from fastapi import APIRouter, Depends, Form, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from typing import Optional
from uuid import UUID
import boto3
import json
import modal
import httpx
from core.api.dependencies import verify_api_key
from core.db.repositories.html_file import HTMLFileRepository
from core.marker import MarkerAPICall
from core.execution.s3 import upload_to_s3
import logging
logger = logging.getLogger(__name__)
from core.models.html_file import HTMLFileCreate

# Create the routers
router_v1 = APIRouter(prefix="/api/v1/convert", tags=["Document Conversion v1"])
router_v2 = APIRouter(prefix="/api/v2/convert", tags=["Document Conversion v2"])
router_v3 = APIRouter(prefix="/api/v3/convert", tags=["Document Conversion v3"])

# Export all routers for server.py to use
routers = [router_v1, router_v2, router_v3]

# Dictionary to store task status
task_statuses = {}

# Constants
DOMAIN_NAME = os.environ.get("STRUCTURA_DOMAIN", "http://localhost:8000")
MARKER_BASE_URL = "https://www.datalab.to/api/v1/marker"

@router_v1.post(
    "",
    summary="Convert a document",
    dependencies=[Depends(verify_api_key)]
)
async def convert_document(
    file: UploadFile = File(...),
    document_type_id: Optional[UUID] = None,
    output_format: str = "json",
    langs: str = None,
    force_ocr: bool = False,
    paginate: bool = False,
    use_llm: bool = True,
    strip_existing_ocr: bool = False,
    disable_image_extraction: bool = False,
    max_pages: int = None
):
    """
    Convert a document with optional parameters.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        marker = MarkerAPICall(tmp_path)
        submit_result = marker.submit_pdf(tmp_path)
        local_url = f"{DOMAIN_NAME}/api/v1/convert/{submit_result['request_id']}"
        return JSONResponse(content={
            "success": True,
            "request_id": submit_result['request_id'],
            "request_check_url": local_url
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@router_v1.get(
    "/{task_id}",
    summary="Get conversion result",
    dependencies=[Depends(verify_api_key)]
)
async def get_conversion_result(
    task_id: str,
    document_type_id: Optional[UUID] = None,
):
    """
    Get the result of a document conversion.
    """
    # Construct the marker API check URL directly
    request_check_url = f"{MARKER_BASE_URL}/{task_id}"
    try:
        marker = MarkerAPICall(pdf_path="dummy.pdf")  # pdf_path not used for polling
        result = marker.poll_result(request_check_url)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_v2.post(
    "",
    summary="Convert a document using Modal",
    dependencies=[Depends(verify_api_key)]
)
async def convert_document_v2(
    file: Optional[UploadFile] = File(None),
    pdf_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    output_format: str = Form("json"),
    use_llm: bool = Form(True),
    user_id: Optional[UUID] = Form(None),  # Make UUID optional
    file_id: Optional[UUID] = Form(None),   # Make UUID optional
    modal_app: str = Form("marker")  # Modal app: "marker" (default), "v31", "v32"
):
    """
    Convert a document using Modal for PDF processing.
    Accepts either a file upload OR a pdf_url parameter.

    Args:
        modal_app: Modal app to use. Options: "marker" (default/production), "v31", "v32"
    """
    # Map shorthand names to full Modal app names
    modal_app_mapping = {
        "marker": "marker",
        "v31": "marker-v31",
        "v32": "marker-v32",
        # Also accept full names for backwards compatibility
        "marker-v31": "marker-v31",
        "marker-v32": "marker-v32"
    }

    # Validate modal_app parameter
    if modal_app not in modal_app_mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid modal_app. Must be one of: marker, v31, v32"
        )

    # Map to full Modal app name
    modal_app = modal_app_mapping[modal_app]

    # Validate that either file or pdf_url is provided
    if not file and not pdf_url:
        raise HTTPException(
            status_code=422,
            detail="Either 'file' or 'pdf_url' must be provided"
        )

    if file and pdf_url:
        raise HTTPException(
            status_code=422,
            detail="Cannot provide both 'file' and 'pdf_url'. Choose one."
        )

    # Validate file content type if file is provided
    if file and file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    # Check for existing record first to prevent duplicate tasks
    existing_file = None
    if user_id and file_id:
        try:
            repo = HTMLFileRepository()
            existing_file = await repo.get_by_user_and_file_with_status(user_id, file_id)

            if existing_file:
                # Check status of existing file
                if existing_file.status in ["completed", "processing"]:
                    # Return existing task info
                    task_id = existing_file.task_id or str(uuid.uuid4())
                    local_url = f"{DOMAIN_NAME}/api/v2/convert/{task_id}"
                    return JSONResponse(content={
                        "success": True,
                        "request_id": str(task_id),
                        "request_check_url": local_url,
                        "status": existing_file.status,
                        "html_file_id": str(existing_file.id)
                    })
                elif existing_file.status == "failed":
                    # Allow retry by continuing with new task
                    logger.info(f"Retrying failed task for user_id: {user_id}, file_id: {file_id}")
                else:
                    # For pending status, return existing task
                    task_id = existing_file.task_id or str(uuid.uuid4())
                    local_url = f"{DOMAIN_NAME}/api/v2/convert/{task_id}"
                    return JSONResponse(content={
                        "success": True,
                        "request_id": str(task_id),
                        "request_check_url": local_url,
                        "status": existing_file.status,
                        "html_file_id": str(existing_file.id)
                    })
        except Exception as e:
            logger.error(f"Error checking existing file: {str(e)}")
            # Continue with new task if checking fails

    # Generate a task ID
    task_id = str(uuid.uuid4())

    # Store user_id, file_id, and modal_app in task_statuses
    task_statuses[task_id] = {
        "status": "pending",
        "user_id": str(user_id) if user_id else None,
        "file_id": str(file_id) if file_id else None,
        "modal_app": modal_app
    }

    # Create temp file - either from upload or download from URL
    tmp_path = None
    try:
        if file:
            # Handle file upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
        else:
            # Handle PDF URL download
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get("content-type", "")
                if "application/pdf" not in content_type:
                    raise HTTPException(
                        status_code=415,
                        detail=f"URL does not point to a PDF file. Content-Type: {content_type}"
                    )

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download PDF from URL: {str(e)}"
        )
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

    try:
        # Create initial HTML file record with pending status if we have user_id and file_id
        created_file = None
        if user_id and file_id:
            try:
                html_file = HTMLFileCreate(
                    user_id=user_id,
                    file_id=file_id,
                    marker_output={},  # Empty for now
                    status="pending",
                    task_id=UUID(task_id),
                    metadata={"source": "marker"}
                )
                repo = HTMLFileRepository()
                created_file = await repo.create(html_file)
                logger.info(f"Created initial HTML file record with ID: {created_file.id}")
            except Exception as e:
                logger.error(f"Error creating initial HTML file: {str(e)}")
                # Continue with processing even if database record creation fails

        # Upload the file to S3
        bucket_name = "marker-pdf"
        s3_url = upload_to_s3(tmp_path, bucket_name)

        # Append the task_id to the URL so modal function can extract it
        s3_url_with_task_id = f"{s3_url}/{task_id}"

        # Get the Modal function by name and spawn it
        logger.info(f"Using Modal app: {modal_app}")
        fn = modal.Function.from_name(modal_app, "parse_pdf")
        call_id = fn.spawn(s3_url_with_task_id)
        print(f"Modal call ID: {call_id} (app: {modal_app})")

        # Update task status with modal_call_id but preserve user_id and file_id
        task_statuses[task_id].update({
            "status": "processing",
            "modal_call_id": str(call_id)
        })

        # Update database record if we have one
        if created_file:
            try:
                repo = HTMLFileRepository()
                await repo.update_status(
                    html_file_id=created_file.id,
                    status="processing",
                    task_id=UUID(task_id),
                )
            except Exception as e:
                logger.error(f"Error updating database status: {str(e)}")

        # Return response with task ID
        local_url = f"{DOMAIN_NAME}/api/v2/convert/{task_id}"
        return JSONResponse(content={
            "success": True,
            "request_id": task_id,
            "request_check_url": local_url
        })

    except Exception as e:
        # Update task status to failed but preserve user_id and file_id
        task_statuses[task_id].update({
            "status": "failed",
            "error": str(e)
        })

        # Update database record if we have one
        if created_file:
            try:
                repo = HTMLFileRepository()
                await repo.update_status(
                    html_file_id=created_file.id,
                    status="failed",
                    task_id=UUID(task_id)
                )
            except Exception as db_error:
                logger.error(f"Error updating database status to failed: {str(db_error)}")

        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@router_v2.get(
    "/{task_id}",
    summary="Get conversion result from Modal",
    dependencies=[Depends(verify_api_key)]
)
async def get_conversion_result_v2(
    task_id: str
):
    """
    Get the result of a document conversion processed by Modal.
    """
    try:
        # First check if we have a record with this task_id
        repo = HTMLFileRepository()
        existing_file = await repo.get_by_task_id(task_id)

        if existing_file:
            if existing_file.status == "completed":
                return JSONResponse(content={
                    "status": "completed",
                    "result": existing_file.marker_output,
                    "html_file_id": str(existing_file.id)
                })
            elif existing_file.status == "failed":
                return JSONResponse(content={
                    "status": "failed",
                    "error": existing_file.metadata.get("error", "Processing failed") if existing_file.metadata else "Processing failed"
                })
            # For "pending" and "processing" statuses, continue with S3 polling below

        # Continue with S3 polling logic for pending/processing tasks or when no database record exists
        session = boto3.session.Session()
        s3_client = session.client('s3',
                                endpoint_url=os.environ.get("DO_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("DO_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("DO_SECRET_ACCESS_KEY"))

        bucket_name = "pdf-results"

        # Get existing task status first to preserve user_id and file_id
        existing_task_status = task_statuses.get(task_id, {})
        user_id = existing_task_status.get("user_id")
        file_id = existing_task_status.get("file_id")

        # If we have user_id and file_id, check for existing record first
        if user_id and file_id:
            try:
                existing_file = await repo.get_by_user_and_file(
                    UUID(user_id) if isinstance(user_id, str) else user_id,
                    UUID(file_id) if isinstance(file_id, str) else file_id
                )
                if existing_file and existing_file.status == "completed":
                    logger.info(f"Found existing completed HTML file for user_id: {user_id}, file_id: {file_id}")
                    return JSONResponse(content={
                        "status": "completed",
                        "result": existing_file.marker_output,
                        "html_file_id": str(existing_file.id)
                    })
            except Exception as e:
                logger.error(f"Error checking for existing HTML file: {str(e)}")
                # Continue with normal flow if checking fails

        # First, check if the exact task_id exists as key
        try:
            # Try to get the file with exact task_id as key
            response = s3_client.get_object(Bucket=bucket_name, Key=f"{task_id}.json")
            json_content = json.loads(response['Body'].read().decode('utf-8'))

            # Update task status while preserving user_id and file_id
            task_statuses[task_id] = {
                "status": "completed",
                "result_url": f"{os.environ.get('DO_ENDPOINT_URL')}/{bucket_name}/{task_id}.json",
                "user_id": user_id,
                "file_id": file_id
            }

            # If we have user_id and file_id, create or update HTML file
            if user_id and file_id:
                try:
                    # Convert UUIDs if needed
                    user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
                    file_uuid = UUID(file_id) if isinstance(file_id, str) else file_id

                    # Check if we already have a record for this user_id/file_id
                    existing_file = await repo.get_by_user_and_file(user_uuid, file_uuid)

                    if existing_file:
                        # Update existing record status
                        await repo.update_status(
                            html_file_id=existing_file.id,
                            status="completed",
                            task_id=UUID(task_id)
                        )

                        # Update marker_output via repository method
                        metadata = {
                            "task_id": task_id,
                            "marker_task_id": json_content.get("task_id"),
                            "processing_time": json_content.get("processing_time"),
                            "source": json_content.get("source", "marker")
                        }
                        await repo.update_marker_output(
                            html_file_id=existing_file.id,
                            marker_output=json_content,
                            metadata=metadata
                        )

                        return JSONResponse(content={
                            "status": "completed",
                            "result": json_content,
                            "html_file_id": str(existing_file.id)
                        })
                    else:
                        # Create new record
                        metadata = {
                            "task_id": task_id,
                            "marker_task_id": json_content.get("task_id"),
                            "processing_time": json_content.get("processing_time"),
                            "source": json_content.get("source", "marker")
                        }

                        html_file = HTMLFileCreate(
                            user_id=user_uuid,
                            file_id=file_uuid,
                            marker_output=json_content,
                            status="completed",
                            task_id=UUID(task_id),
                            metadata=metadata
                        )

                        created_file = await repo.create(html_file)

                        return JSONResponse(content={
                            "status": "completed",
                            "result": json_content,
                            "html_file_id": str(created_file.id)
                        })

                except Exception as save_error:
                    if "duplicate key value" in str(save_error):
                        # If we hit a duplicate key error, try to fetch the existing record
                        try:
                            existing_file = await repo.get_by_user_and_file(user_uuid, file_uuid)
                            if existing_file:
                                return JSONResponse(content={
                                    "status": "completed",
                                    "result": existing_file.marker_output,
                                    "html_file_id": str(existing_file.id)
                                })
                        except Exception as fetch_error:
                            logger.error(f"Error fetching existing record after duplicate key error: {str(fetch_error)}")

                    return JSONResponse(content={
                        "status": "completed",
                        "result": json_content,
                        "error": f"Failed to save HTML file: {str(save_error)}"
                    })

            # If we don't have user_id and file_id, return just the result
            return JSONResponse(content={
                "status": "completed",
                "result": json_content
            })

        except s3_client.exceptions.NoSuchKey:
            # If key isn't found, preserve user_id and file_id in processing status
            task_statuses[task_id] = {
                "status": "processing",
                "user_id": user_id,
                "file_id": file_id
            }

            # Return appropriate status based on what we found in database
            if existing_file:
                return JSONResponse(content={
                    "status": existing_file.status,
                    "html_file_id": str(existing_file.id)
                })
            else:
                return JSONResponse(content={"status": "processing"})

    except Exception as e:
        return JSONResponse(content={"status": "processing"})

@router_v2.get(
    "/html/{user_id}/{file_id}",
    summary="Get HTML file and marker output by user_id and file_id",
    dependencies=[Depends(verify_api_key)]
)
async def get_html_file(
    user_id: UUID,
    file_id: UUID
):
    """
    Get HTML file and marker output for a specific user_id and file_id combination.
    """
    try:
        repo = HTMLFileRepository()
        html_file = await repo.get_by_user_and_file(user_id, file_id)

        if not html_file:
            return JSONResponse(
                status_code=404,
                content={"error": "HTML file not found"}
            )

        return JSONResponse(content={
            "id": str(html_file.id),
            "marker_output": html_file.marker_output,
            "metadata": html_file.metadata
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )


@router_v3.post(
    "",
    summary="Convert PDF to HTML using Gemini 2.5 Pro directly",
    dependencies=[Depends(verify_api_key)]
)
async def convert_document_v3(
    file: Optional[UploadFile] = File(None),
    pdf_url: Optional[str] = Form(None),
    user_id: Optional[UUID] = Form(None),
    file_id: Optional[UUID] = Form(None)
):
    """
    Convert a PDF document to HTML using Gemini 2.5 Pro directly.

    This endpoint bypasses Marker OCR and sends the PDF pages directly to Gemini
    for HTML extraction, providing higher quality output at the cost of speed.

    Args:
        file: PDF file upload (mutually exclusive with pdf_url)
        pdf_url: URL to PDF file (mutually exclusive with file)
        user_id: Optional user ID for tracking
        file_id: Optional file ID for tracking

    Returns:
        JSON response with HTML content for each page
    """
    from google import genai
    from google.genai import types
    from pdf2image import convert_from_path
    from io import BytesIO
    import PIL.Image

    # Validate input
    if not file and not pdf_url:
        raise HTTPException(
            status_code=422,
            detail="Either 'file' or 'pdf_url' must be provided"
        )

    if file and pdf_url:
        raise HTTPException(
            status_code=422,
            detail="Cannot provide both 'file' and 'pdf_url'. Choose one."
        )

    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY not configured"
        )

    tmp_path = None

    try:
        # Download or read the PDF file to temp file
        if file:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=415, detail="Only PDF files are supported.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
        else:
            # Download from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "application/pdf" not in content_type:
                    raise HTTPException(
                        status_code=415,
                        detail=f"URL does not point to a PDF file. Content-Type: {content_type}"
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

        logger.info(f"Converting PDF to images for page-by-page processing...")

        # Convert PDF to images (one per page)
        page_images = convert_from_path(tmp_path, dpi=200)
        total_pages = len(page_images)
        logger.info(f"Converted {total_pages} pages to images")

        # Helper function to process a single page with retries
        def process_page_with_retry(page_num: int, page_image: PIL.Image.Image, max_retries: int = 10) -> tuple[int, str]:
            """Process a single page with Gemini, with retry logic"""
            import time

            # Initialize Gemini client (one per page for thread safety)
            client = genai.Client(api_key=api_key)

            # Convert PIL image to bytes
            image_bytes = BytesIO()
            page_image.save(image_bytes, format="PNG")

            image_part = types.Part.from_bytes(
                data=image_bytes.getvalue(),
                mime_type="image/png"
            )

            prompt = f"""You are an expert document parser. Parse this page (page {page_num}) from a PDF document and convert it to clean, semantic HTML.

IMPORTANT INSTRUCTIONS:
1. Extract ALL text content exactly as it appears
2. Preserve document structure using semantic HTML tags (h1-h6, p, ul, ol, table, etc.)
3. For mathematical expressions, use LaTeX notation:
   - Inline math: $expression$
   - Display math: $$expression$$
4. For tables, use proper HTML table structure with <table>, <thead>, <tbody>, <tr>, <th>, <td>
5. Maintain the visual hierarchy and relationships between elements
6. Do NOT add any wrapper elements or explanatory text
7. Do NOT use markdown code blocks - return ONLY the HTML
8. Return ONLY the HTML content for this page

Parse the page and return the HTML:"""

            for attempt in range(max_retries):
                try:
                    logger.info(f"Processing page {page_num}/{total_pages} (attempt {attempt + 1})...")

                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=[image_part, prompt],
                        config=types.GenerateContentConfig(
                            temperature=0,
                        )
                    )

                    page_html = response.text

                    # Remove markdown code blocks if present
                    if page_html.strip().startswith('```'):
                        lines = page_html.strip().split('\n')
                        if lines[0].startswith('```'):
                            lines = lines[1:]
                        if lines and lines[-1].startswith('```'):
                            lines = lines[:-1]
                        page_html = '\n'.join(lines)

                    logger.info(f"Page {page_num} completed successfully")
                    return page_num, page_html

                except Exception as e:
                    error_str = str(e)
                    logger.warning(f"Page {page_num} attempt {attempt + 1} failed: {error_str}")

                    # Retry on 503, rate limits, or transient errors
                    if attempt < max_retries - 1:
                        if "503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str or "rate" in error_str.lower():
                            wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                            logger.info(f"Retrying page {page_num} in {wait_time}s...")
                            time.sleep(wait_time)
                            continue

                    # Raise exception after all retries exhausted
                    logger.error(f"Page {page_num} failed after {max_retries} attempts: {e}")
                    raise Exception(f"Page {page_num} failed after {max_retries} attempts: {str(e)}")

            raise Exception(f"Page {page_num} failed: Maximum retries exceeded")

        # Process all pages in parallel using ThreadPoolExecutor
        import concurrent.futures

        logger.info(f"Processing {total_pages} pages in parallel with Gemini 2.5 Pro...")

        pages_dict = {}
        max_workers = min(5, total_pages)  # Use up to 5 parallel workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages for processing
            futures = []
            for page_num, page_image in enumerate(page_images, start=1):
                future = executor.submit(process_page_with_retry, page_num, page_image)
                futures.append(future)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                page_num, page_html = future.result()
                pages_dict[page_num] = page_html

        # Compile pages in correct order as array
        pages_array = [
            {"page": i, "html": pages_dict[i]}
            for i in range(1, total_pages + 1)
        ]

        logger.info(f"All {total_pages} pages processed successfully")

        # Prepare response
        result = {
            "status": "success",
            "pages": pages_array,
            "total_pages": total_pages
        }

        logger.info(f"Conversion complete")

        return JSONResponse(content=result)

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download PDF from URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in v3 conversion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")
