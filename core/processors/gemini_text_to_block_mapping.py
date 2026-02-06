"""
Create a mapping from Gemini-corrected text spans to original block IDs.

This allows bidirectional click mapping without distributing corrections back to blocks.
"""

from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
import difflib


def extract_text_with_positions(html: str) -> List[Tuple[int, int, str]]:
    """
    Extract text from HTML with character positions.

    Returns list of (start_pos, end_pos, text) tuples.
    """
    soup = BeautifulSoup(html, 'html.parser')

    positions = []
    current_pos = 0

    # Get all text nodes
    for element in soup.descendants:
        if isinstance(element, str):
            text = element.strip()
            if text:
                start = current_pos
                end = current_pos + len(text)
                positions.append((start, end, text))
                current_pos = end + 1  # +1 for space

    return positions


def create_text_to_block_mapping(
    corrected_html: str,
    original_blocks: List[Dict]
) -> Dict[int, str]:
    """
    Create mapping from character positions in corrected HTML to block IDs.

    Args:
        corrected_html: The full Gemini-corrected HTML for the page
        original_blocks: List of original blocks with IDs and HTML

    Returns:
        Dictionary mapping char_position -> block_id
    """
    # Extract text spans from corrected HTML
    corrected_spans = extract_text_with_positions(corrected_html)

    # Extract text from each original block
    block_texts = []
    for block in original_blocks:
        html = block.get('html', '')
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text().strip()
        if text:
            block_texts.append({
                'id': block.get('id'),
                'text': text,
                'html': html
            })

    # Create mapping
    mapping = {}

    for start_pos, end_pos, span_text in corrected_spans:
        # Find which original block this text came from
        best_match_id = None
        best_ratio = 0.0

        for block_data in block_texts:
            ratio = difflib.SequenceMatcher(
                None,
                span_text.lower(),
                block_data['text'].lower()
            ).ratio()

            if ratio > best_ratio and ratio > 0.5:  # Minimum 50% match
                best_ratio = ratio
                best_match_id = block_data['id']

        # Map all character positions in this span to the block ID
        if best_match_id:
            for pos in range(start_pos, end_pos + 1):
                mapping[pos] = best_match_id

    return mapping


def get_block_id_for_text_selection(
    mapping: Dict[int, str],
    start_pos: int,
    end_pos: int
) -> str:
    """
    Get the block ID for a text selection in the corrected HTML.

    Args:
        mapping: Character position -> block ID mapping
        start_pos: Start character position
        end_pos: End character position

    Returns:
        Block ID that covers most of the selection, or None
    """
    # Count which block ID appears most in the selection range
    block_counts = {}

    for pos in range(start_pos, end_pos + 1):
        block_id = mapping.get(pos)
        if block_id:
            block_counts[block_id] = block_counts.get(block_id, 0) + 1

    if not block_counts:
        return None

    # Return the block with the most characters in the selection
    return max(block_counts.items(), key=lambda x: x[1])[0]


def add_data_attributes_to_html(
    html: str,
    mapping: Dict[int, str]
) -> str:
    """
    Add data-block-id attributes to HTML elements based on mapping.

    This allows click handlers to identify which block an element belongs to.
    """
    soup = BeautifulSoup(html, 'html.parser')

    current_pos = 0

    # Add data attributes to each element
    for element in soup.find_all(recursive=True):
        if element.name and element.string:
            text = element.string.strip()
            if text:
                # Find block ID for this element's position
                block_id = mapping.get(current_pos)
                if block_id:
                    element['data-block-id'] = block_id

                current_pos += len(text) + 1

    return str(soup)


def create_gemini_page_with_mapping(page: Dict, corrected_html: str) -> Dict:
    """
    Create page structure with Gemini HTML and separate mapping.

    This is the new approach that doesn't distribute corrections back to blocks.

    Returns:
        {
            'id': page_id,
            'gemini_html': corrected_html_with_data_attrs,
            'original_blocks': [...],  # Preserved for bbox info
            'text_to_block_mapping': {...}  # For click mapping
        }
    """
    original_blocks = page.get('children', [])

    # Create character position -> block ID mapping
    mapping = create_text_to_block_mapping(corrected_html, original_blocks)

    # Add data-block-id attributes to the HTML
    html_with_attrs = add_data_attributes_to_html(corrected_html, mapping)

    return {
        'id': page.get('id'),
        'gemini_html': html_with_attrs,
        'original_blocks': original_blocks,  # Keep for bbox info
        'text_to_block_mapping': mapping,
        '_approach': 'direct_render_with_mapping'
    }
