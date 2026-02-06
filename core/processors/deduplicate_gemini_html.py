"""
Deduplicate content in Gemini-corrected HTML.

Gemini sometimes returns duplicate content in its corrections.
This module detects and removes such duplicates.
"""

from bs4 import BeautifulSoup
from typing import List, Set
import difflib


def extract_text_from_element(element) -> str:
    """Extract clean text from an HTML element."""
    return element.get_text().strip()


def are_elements_duplicate(elem1, elem2, threshold: float = 0.9) -> bool:
    """
    Check if two HTML elements are duplicates based on text similarity.

    Args:
        elem1: First BeautifulSoup element
        elem2: Second BeautifulSoup element
        threshold: Similarity threshold (0-1)

    Returns:
        True if elements are duplicates
    """
    text1 = extract_text_from_element(elem1)
    text2 = extract_text_from_element(elem2)

    if not text1 or not text2:
        return False

    # Use sequence matcher to compare
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()

    return ratio >= threshold


def deduplicate_html(html: str) -> str:
    """
    Remove duplicate content from Gemini HTML.

    Gemini sometimes duplicates list items or entire sections.
    This function identifies and removes such duplicates.

    Strategy:
    1. Parse HTML into elements
    2. Track seen content using text similarity
    3. Remove elements that are duplicates of earlier ones
    4. Preserve the first occurrence

    Args:
        html: Gemini-corrected HTML with potential duplicates

    Returns:
        Deduplicated HTML
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find all top-level list structures (ol, ul)
    top_level_lists = soup.find_all(['ol', 'ul'], recursive=False)

    # Track elements we've seen
    seen_elements = []
    elements_to_remove = []

    for list_elem in top_level_lists:
        # Get all direct children (list items)
        items = list_elem.find_all('li', recursive=False)

        for item in items:
            # Check if this item is a duplicate of any seen item
            is_duplicate = False

            for seen_item in seen_elements:
                if are_elements_duplicate(item, seen_item, threshold=0.9):
                    is_duplicate = True
                    break

            if is_duplicate:
                # Mark entire parent list for removal if all its items are duplicates
                elements_to_remove.append(list_elem)
                break
            else:
                # Add to seen elements
                seen_elements.append(item)

    # Remove duplicate elements
    for elem in elements_to_remove:
        elem.decompose()

    return str(soup)


def deduplicate_gemini_page(page: dict) -> dict:
    """
    Deduplicate Gemini-corrected HTML in a page structure.

    For pages with GeminiCorrected blocks, removes duplicate content.

    Args:
        page: Page dict with children containing GeminiCorrected block

    Returns:
        Page dict with deduplicated HTML
    """
    if not page.get('children'):
        return page

    gemini_block = page['children'][0]

    if 'GeminiCorrected' not in gemini_block.get('id', ''):
        return page

    # Get original HTML
    original_html = gemini_block.get('html', '')

    # Deduplicate
    deduplicated_html = deduplicate_html(original_html)

    # Update block
    gemini_block['html'] = deduplicated_html
    gemini_block['_deduplicated'] = True
    gemini_block['_original_html_length'] = len(original_html)
    gemini_block['_deduplicated_html_length'] = len(deduplicated_html)

    return page
