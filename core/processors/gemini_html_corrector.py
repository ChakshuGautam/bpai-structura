"""
Post-processor to correct HTML output using Gemini by comparing with page images.
"""
import os
import re
import time
import logging
from io import BytesIO
from typing import Dict, List
from pdf2image import convert_from_path
import PIL.Image
from google import genai
from google.genai import types
import concurrent.futures

logger = logging.getLogger(__name__)


def convert_math_tags(html: str) -> str:
    """Convert <math> tags to MathJax delimiters"""
    # Handle block display math
    html = re.sub(
        r'<math display="block">(.*?)</math>',
        lambda m: f'$${m.group(1)}$$',
        html,
        flags=re.DOTALL
    )

    # Handle inline math (with or without display="inline")
    html = re.sub(
        r'<math(?:\s+display="inline")?>(.*?)</math>',
        lambda m: f'${m.group(1)}$',
        html,
        flags=re.DOTALL
    )

    return html


def render_block(block: dict, depth: int = 0) -> str:
    """Recursively render blocks to HTML"""
    html = block.get('html', '')
    if html:
        html = convert_math_tags(html)

    result = html

    # Render children
    children = block.get('children')
    if children:
        for child in children:
            result += "\n" + render_block(child, depth + 1)

    return result


def get_page_html(page_data: dict) -> str:
    """Get HTML for a single page"""
    html_parts = []
    for block in page_data.get('children', []):
        html_parts.append(render_block(block))
    return "\n".join(html_parts)


def process_page_with_gemini(
    page_num: int,
    page_image: PIL.Image.Image,
    page_html: str,
    api_key: str,
    max_retries: int = 3
) -> tuple[int, str]:
    """Process a single page with Gemini to correct HTML"""
    logger.info(f"Processing page {page_num} with Gemini...")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Convert PIL image to bytes
    image_bytes = BytesIO()
    page_image.save(image_bytes, format="PNG")

    image_part = types.Part.from_bytes(
        data=image_bytes.getvalue(),
        mime_type="image/png"
    )

    prompt = f"""You are a document extraction expert. You are given:
1. An image of page {page_num} from a PDF document
2. The HTML content that was extracted from this page

Your task is to:
1. Carefully review the PDF image and the extracted HTML
2. Fix any errors in the HTML extraction, including:
   - Missing content
   - Incorrect math expressions (especially superscripts like 17^{{th}} should remain as LaTeX)
   - OCR errors in text
   - Structural issues
3. Return ONLY the corrected HTML content for this page, preserving the existing HTML structure

IMPORTANT:
- Keep all math in LaTeX format with $ for inline and $$ for display math
- Do NOT wrap your response in markdown code blocks
- Return ONLY the HTML content, nothing else

Here is the extracted HTML for page {page_num}:

{page_html}

Return the corrected HTML:"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0,
                )
            )

            corrected_html = response.text

            # Remove markdown code blocks if present
            if corrected_html.strip().startswith('```'):
                lines = corrected_html.strip().split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                corrected_html = '\n'.join(lines)

            logger.info(f"Page {page_num} completed")
            return page_num, corrected_html

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    logger.warning(
                        f"Page {page_num} - 503 error, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Page {page_num} - Failed after {max_retries} attempts: {e}")
            else:
                logger.error(f"Error processing page {page_num}: {e}")
            return page_num, page_html  # Return original on error

    return page_num, page_html


def update_page_with_corrected_html(page: dict, corrected_html: str) -> None:
    """
    Update the page structure to use corrected HTML.

    Since Gemini returns HTML for the entire page and matching individual blocks
    is complex, we create a single block with all the corrected HTML for the page.
    """
    # Store original blocks as backup (optional, for debugging)
    if page.get('children'):
        page['_original_blocks'] = page['children'].copy()

    # Replace page children with a single block containing corrected HTML
    page['children'] = [{
        'id': f"{page.get('id', 'page')}/GeminiCorrected",
        'block_type': 'Text',
        'html': corrected_html,
        'polygon': page.get('polygon', []),
        'structure': None,
        'source': page.get('source'),
        'children': None
    }]


def correct_html_with_gemini(
    pdf_path: str,
    json_data: dict,
    api_key: str = None,
    dpi: int = 100,
    max_workers: int = 5
) -> dict:
    """
    Correct HTML in JSON data by comparing with PDF page images using Gemini.

    Args:
        pdf_path: Path to the PDF file
        json_data: JSON data with document structure and HTML
        api_key: Gemini API key (if None, uses GOOGLE_API_KEY env var)
        dpi: DPI for PDF to image conversion (lower = faster but less detail)
        max_workers: Number of parallel workers

    Returns:
        Updated JSON data with corrected HTML in a separate field per page
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not set, skipping HTML correction")
            return json_data

    logger.info(f"Converting PDF to images at {dpi} DPI...")
    try:
        page_images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Converted {len(page_images)} pages")
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        return json_data

    pages = json_data.get('children', [])
    if len(page_images) != len(pages):
        logger.warning(f"PDF has {len(page_images)} pages but JSON has {len(pages)} pages")

    logger.info("Processing pages with Gemini...")
    corrected_pages = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (page_image, page_data) in enumerate(zip(page_images, pages)):
            page_num = idx + 1
            page_html = get_page_html(page_data)
            future = executor.submit(
                process_page_with_gemini,
                page_num,
                page_image,
                page_html,
                api_key
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            page_num, corrected_html = future.result()
            corrected_pages[page_num] = corrected_html

    # Replace page blocks with corrected HTML
    # This maintains the same JSON structure, just with corrected content
    for idx, page in enumerate(pages):
        page_num = idx + 1
        if page_num in corrected_pages:
            update_page_with_corrected_html(page, corrected_pages[page_num])

    logger.info("HTML correction complete")
    return json_data
