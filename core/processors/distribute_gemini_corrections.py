"""
Distribute Gemini-corrected HTML back to original child blocks.

This solves the problem where GeminiCorrected creates a single merged block,
breaking the click-to-scroll mapping between PDF and HTML viewers.
"""
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import difflib


def extract_text_content(html: str) -> str:
    """Extract plain text from HTML for comparison"""
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(strip=True, separator=' ')


def find_best_match(
    target_text: str,
    candidates: List[str],
    threshold: float = 0.6
) -> Optional[int]:
    """
    Find the best matching candidate for target text using fuzzy matching.

    Returns index of best match, or None if no match above threshold.
    """
    if not target_text or not candidates:
        return None

    best_ratio = 0
    best_idx = None

    for idx, candidate in enumerate(candidates):
        ratio = difflib.SequenceMatcher(None, target_text, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    return best_idx if best_ratio >= threshold else None


def split_html_into_segments(html: str) -> List[str]:
    """
    Split HTML into segments that likely correspond to original blocks.

    Strategy:
    1. Split by top-level HTML elements (h1-h6, p, div, table, etc.)
    2. Each segment should be a complete HTML element
    """
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    segments = []

    # Get direct children of the body (or root if no body)
    root = soup.find('body') if soup.find('body') else soup

    for element in root.children:
        if element.name:  # Skip text nodes
            segments.append(str(element))

    return segments


def distribute_corrections_to_blocks(
    corrected_html: str,
    original_blocks: List[Dict]
) -> List[Dict]:
    """
    Distribute corrected HTML back to original blocks.

    Args:
        corrected_html: The corrected HTML from Gemini (entire page)
        original_blocks: List of original child blocks from the page

    Returns:
        Updated blocks with corrected HTML while preserving structure
    """
    # Split corrected HTML into segments
    corrected_segments = split_html_into_segments(corrected_html)

    # Extract text from original blocks for matching
    original_texts = [extract_text_content(block.get('html', '')) for block in original_blocks]
    corrected_texts = [extract_text_content(seg) for seg in corrected_segments]

    # Track which corrected segments have been used
    used_segments = set()

    # Create updated blocks
    updated_blocks = []

    for block_idx, block in enumerate(original_blocks):
        updated_block = block.copy()
        original_text = original_texts[block_idx]

        # Skip empty blocks - keep them as is
        if not original_text.strip():
            updated_blocks.append(updated_block)
            continue

        # Find best matching corrected segment
        # Only search in unused segments
        available_indices = [i for i in range(len(corrected_segments)) if i not in used_segments]
        available_texts = [corrected_texts[i] for i in available_indices]

        if available_texts:
            # Use lower threshold for better matching (0.3 instead of 0.5)
            # Also try partial matching
            match_idx = find_best_match(original_text, available_texts, threshold=0.3)

            if match_idx is not None:
                actual_idx = available_indices[match_idx]
                # Update the block's HTML with corrected version
                updated_block['html'] = corrected_segments[actual_idx]
                updated_block['_gemini_corrected'] = True
                updated_block['_original_html'] = block.get('html')
                used_segments.add(actual_idx)
            else:
                # No good match found, keep original but mark it
                updated_block['_gemini_correction_failed'] = True
        else:
            # No more corrected segments available
            updated_block['_no_correction_available'] = True

        updated_blocks.append(updated_block)

    # NOTE: We intentionally do NOT add unused segments as new blocks
    # because they are usually just failed matches, not truly new content.
    # This prevents duplicates from appearing in the output.
    #
    # If Gemini genuinely added new content, it will be lost, but this is
    # preferable to showing duplicate content to users.
    #
    # Future improvement: Use better matching algorithm to reduce false negatives

    return updated_blocks


def update_page_with_distributed_corrections(page: Dict, corrected_html: str) -> None:
    """
    Update page structure by distributing Gemini corrections to original blocks.

    This replaces the old approach of creating a single GeminiCorrected block.
    """
    # Get original blocks
    original_blocks = page.get('children', [])

    if not original_blocks:
        # No blocks to update, create single corrected block as fallback
        page['children'] = [{
            'id': f"{page.get('id', 'page')}/GeminiCorrected",
            'block_type': 'Text',
            'html': corrected_html,
            'polygon': page.get('polygon', []),
            'structure': None,
            'source': page.get('source'),
            'children': None
        }]
        return

    # Store original blocks as backup
    page['_original_blocks_backup'] = [block.copy() for block in original_blocks]

    # Distribute corrections
    updated_blocks = distribute_corrections_to_blocks(corrected_html, original_blocks)

    # Update page children
    page['children'] = updated_blocks

    # Add metadata
    page['_gemini_correction_applied'] = True


# Example usage
if __name__ == "__main__":
    # Example: Original page structure
    example_page = {
        'id': '/page/0/Page/26',
        'block_type': 'Page',
        'children': [
            {
                'id': '/page/0/SectionHeader/0',
                'block_type': 'SectionHeader',
                'html': '<h1>Study 1 for HB-GT-101</h1>',
                'polygon': [[0, 0], [100, 0], [100, 20], [0, 20]]
            },
            {
                'id': '/page/0/Text/1',
                'block_type': 'Text',
                'html': '<p>Date: 17th July 2025</p>',
                'polygon': [[0, 25], [100, 25], [100, 40], [0, 40]]
            },
            {
                'id': '/page/0/Text/2',
                'block_type': 'Text',
                'html': '<p>Approved By: Dr. U.P. Shukla</p>',
                'polygon': [[0, 45], [100, 45], [100, 60], [0, 60]]
            }
        ]
    }

    # Example: Gemini's corrected HTML (entire page)
    corrected_html = """
    <h1>Study 1 for HB-GT-101</h1>
    <p>Date: $17^{th}$ July 2025</p>
    <p>Approved By: Dr. U.P. Shukla - Process Development Manager</p>
    """

    # Apply distributed corrections
    update_page_with_distributed_corrections(example_page, corrected_html)

    # Print result
    print("Updated page structure:")
    for block in example_page['children']:
        print(f"\nBlock ID: {block['id']}")
        print(f"HTML: {block['html']}")
        print(f"Gemini corrected: {block.get('_gemini_corrected', False)}")
