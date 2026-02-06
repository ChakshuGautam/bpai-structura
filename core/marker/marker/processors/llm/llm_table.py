from typing import Annotated, List, Tuple

from bs4 import BeautifulSoup
from PIL import Image
from marker.logger import get_logger
from pydantic import BaseModel

from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block, TableCell, Table
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox

logger = get_logger()


class LLMTableProcessor(BaseLLMComplexBlockProcessor):
    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Table, BlockTypes.TableOfContents)
    max_rows_per_batch: Annotated[
        int,
        "If the table has more rows than this, chunk the table. (LLMs can be inaccurate with a lot of rows)",
    ] = 60
    max_table_rows: Annotated[
        int,
        "The maximum number of rows in a table to process with the LLM processor.  Beyond this will be skipped.",
    ] = 175
    table_image_expansion_ratio: Annotated[
        float,
        "The ratio to expand the image by when cropping.",
    ] = 0
    rotation_max_wh_ratio: Annotated[
        float,
        "The maximum width/height ratio for table cells for a table to be considered rotated.",
    ] = 0.6
    table_rewriting_prompt: Annotated[
        str,
        "The prompt to use for rewriting text.",
        "Default is a string containing the Gemini rewriting prompt.",
    ] = """You are a text correction expert specializing in accurately reproducing text from images.
You will receive an image and an html representation of the table in the image.
Your task is to correct any errors in the html representation.  The html representation should be as faithful to the original table image as possible.  The table image may be rotated, but ensure the html representation is not rotated.  Make sure to include HTML for the full table, including the opening and closing table tags.

**Important** if the input HTML is too different from the what you think is correct, just ignore the input html and return the HTML based on what you think is correct.
Rewrite rather than correct in cases of extreme differences.

Some guidelines:
- Reproduce the original values from the image as faithfully as possible.  
- There may be stray characters in the html representation that don't match the image - fix these.
- Ensure column headers match the correct column values.
- If you see any inline math in a table cell, fence it with the <math> tag.  Block math should be fenced with <math display="block">.
- Replace any images in table cells with a description, like "Image: [description]".
- Only use the tags th, td, tr, br, span, sup, sub, i, b, math, and table.  Only use the attributes display, style, colspan, and rowspan if necessary.  You can use br to break up text lines in cells.
- Make sure the columns and rows match the image faithfully, and are easily readable and interpretable by a human.

**Instructions:**
1. Carefully examine the provided text block image.
2. Analyze the html representation of the table.
3. Write a comparison of the image and the html representation, paying special attention to the column headers matching the correct column values.
4. If the html representation is completely correct, or you cannot read the image properly, then write "No corrections needed."  If the html representation has errors, generate the corrected html representation.  Output only either the corrected html representation or "No corrections needed."
**Example:**
Input:
```html
<table>
    <tr>
        <th>First Name</th>
        <th>Last Name</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>John</td>
        <td>Doe</td>
        <td>25</td>
    </tr>
</table>
```
Output:
comparison: The image shows a table with 2 rows and 3 columns.  The text and formatting of the html table matches the image.  The column headers match the correct column values.
```html
No corrections needed.
```
**Input:**
```html
{block_html}
```


"""

    def handle_image_rotation(self, children: List[TableCell], image: Image.Image):
        ratios = [c.polygon.width / c.polygon.height for c in children]
        if len(ratios) < 2:
            return image

        is_rotated = all([r < self.rotation_max_wh_ratio for r in ratios])
        if not is_rotated:
            return image

        first_col_id = min([c.col_id for c in children])
        first_col = [c for c in children if c.col_id == first_col_id]
        first_col_cell = first_col[0]

        last_col_id = max([c.col_id for c in children])
        if last_col_id == first_col_id:
            return image

        last_col_cell = [c for c in children if c.col_id == last_col_id][0]
        cell_diff = first_col_cell.polygon.y_start - last_col_cell.polygon.y_start
        if cell_diff == 0:
            return image

        if cell_diff > 0:
            return image.rotate(270, expand=True)
        else:
            return image.rotate(90, expand=True)

    def process_rewriting(self, document: Document, page: PageGroup, block: Table):
        children: List[TableCell] = block.contained_blocks(
            document, (BlockTypes.TableCell,)
        )
        if not children:
            # Happens if table/form processors didn't run
            return

        # LLMs don't handle tables with a lot of rows very well
        unique_rows = set([cell.row_id for cell in children])
        row_count = len(unique_rows)
        row_idxs = sorted(list(unique_rows))

        if row_count > self.max_table_rows:
            return

        # Inference by chunk to handle long tables better
        parsed_cells = []
        row_shift = 0
        block_image = self.extract_image(document, block)
        block_rescaled_bbox = block.polygon.rescale(
            page.polygon.size, page.get_image(highres=True).size
        ).bbox
        for i in range(0, row_count, self.max_rows_per_batch):
            batch_row_idxs = row_idxs[i : i + self.max_rows_per_batch]
            batch_cells = [cell for cell in children if cell.row_id in batch_row_idxs]
            batch_cell_bboxes = [
                cell.polygon.rescale(
                    page.polygon.size, page.get_image(highres=True).size
                ).bbox
                for cell in batch_cells
            ]
            # bbox relative to the block
            batch_bbox = [
                min([bbox[0] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[0],
                min([bbox[1] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[1],
                max([bbox[2] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[0],
                max([bbox[3] for bbox in batch_cell_bboxes]) - block_rescaled_bbox[1],
            ]
            if i == 0:
                # Ensure first image starts from the beginning
                batch_bbox[0] = 0
                batch_bbox[1] = 0
            elif i > row_count - self.max_rows_per_batch + 1:
                # Ensure final image grabs the entire height and width
                batch_bbox[2] = block_image.size[0]
                batch_bbox[3] = block_image.size[1]

            batch_image = block_image.crop(batch_bbox)
            block_html = block.format_cells(document, [], batch_cells)
            batch_image = self.handle_image_rotation(batch_cells, batch_image)
            batch_parsed_cells = self.rewrite_single_chunk(
                page, block, block_html, batch_cells, batch_image
            )
            if batch_parsed_cells is None:
                print("no changes - Error occurred or no corrections needed")
                return  # Error occurred or no corrections needed

            for cell in batch_parsed_cells:
                cell.row_id += row_shift
                parsed_cells.append(cell)
            row_shift += max([cell.row_id for cell in batch_parsed_cells])

        block.structure = []
        print("Before adding to the page/block", len(parsed_cells))
        for cell in parsed_cells:
            page.add_full_block(cell)
            block.add_structure(cell)

    def rewrite_single_chunk(
        self,
        page: PageGroup,
        block: Block,
        block_html: str,
        children: List[TableCell],
        image: Image.Image,
    ):
        prompt = self.table_rewriting_prompt.replace("{block_html}", block_html)
        
        self.llm_service.timeout = 300
        
        response = self.llm_service(prompt, image, block, TableSchema)
        
        if not response or "corrected_html" not in response:
            block.update_metadata(llm_error_count=1)
            return

        corrected_html = response["corrected_html"]
        
        print("Page Block", block)
        
        print("Original HTML", block_html)
        
        print("Block", block.block_id)
        print("Corrected HTML", response["corrected_html"])

        # The original table is okay
        if "no corrections" in corrected_html.lower():
            print("No Corrections found in Table")
            return

        corrected_html = corrected_html.strip().lstrip("```html").rstrip("```").strip()
        parsed_cells = self.parse_html_table(corrected_html, block, page, children)
        
        print("Parsed Cells", parsed_cells)
        
        if len(parsed_cells) <= 1:
            block.update_metadata(llm_error_count=1)
            print("No Parsed Cells", len(parsed_cells))
            return

        if not corrected_html.endswith("</table>"):
            print("Wrongly formed table - no </table>")
            block.update_metadata(llm_error_count=1)
            return

        # parsed_cell_text = "".join([cell.text for cell in parsed_cells])
        # orig_cell_text = "".join([cell.text for cell in children])
        # Potentially a partial response
        # print("Parsed Table size", len(parsed_cells))
        # if len(parsed_cell_text) < len(orig_cell_text) * 0.5:
        #     print("LLM Error - Partial Response")
        #     block.update_metadata(llm_error_count=1)
        #     return

        return parsed_cells

    @staticmethod
    def get_cell_text(element, keep_tags=("br", "i", "b", "span", "math")) -> str:
        for tag in element.find_all(True):
            if tag.name not in keep_tags:
                tag.unwrap()
        return element.decode_contents()

    def parse_html_table(
        self, html_text: str, block: Block, page: PageGroup, original_children: List[TableCell]
    ) -> List[TableCell]:
        soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        # Initialize grid
        rows = table.find_all("tr")
        cells = []
        
        # Track usage of original cells to handle merging/splitting scenarios
        used_cells = {}

        # Find maximum number of columns in colspan-aware way
        max_cols = 0
        for row in rows:
            row_tds = row.find_all(["td", "th"])
            curr_cols = 0
            for cell in row_tds:
                colspan = int(cell.get("colspan", 1))
                curr_cols += colspan
            if curr_cols > max_cols:
                max_cols = curr_cols

        grid = [[True] * max_cols for _ in range(len(rows))]

        for i, row in enumerate(rows):
            cur_col = 0
            row_cells = row.find_all(["td", "th"])
            for j, cell in enumerate(row_cells):
                while cur_col < max_cols and not grid[i][cur_col]:
                    cur_col += 1

                if cur_col >= max_cols:
                    logger.info("Table parsing warning: too many columns found")
                    break

                cell_text = self.get_cell_text(cell).strip()
                rowspan = min(int(cell.get("rowspan", 1)), len(rows) - i)
                colspan = min(int(cell.get("colspan", 1)), max_cols - cur_col)
                cell_rows = list(range(i, i + rowspan))
                cell_cols = list(range(cur_col, cur_col + colspan))
                
                print(f"Processing cell at Row: {i}, Col: {cur_col}, ColSpan: {colspan}, RowSpan: {rowspan}, Text: '{cell_text}'")

                if colspan == 0 or rowspan == 0:
                    logger.info("Table parsing issue: invalid colspan or rowspan")
                    continue

                for r in cell_rows:
                    for c in cell_cols:
                        grid[r][c] = False
                
                # Try to find a matching polygon from original cells with improved matching
                matched_polygon = find_closest_mapped_bbox(original_children, i, cur_col, cell_text, used_cells)
                
                if matched_polygon:
                    # Use the matched original cell's polygon
                    cell_polygon = matched_polygon
                    print(f"  -> Mapped to original cell (usage count: {max(used_cells.values()) if used_cells else 0})")
                else:
                    # Fallback to creating a new polygon (but with better proportions)
                    # Calculate proportional dimensions based on the table's overall size
                    table_width = block.polygon.width
                    table_height = block.polygon.height
                    
                    col_width = table_width / max(max_cols, 1)
                    row_height = table_height / max(len(rows), 1)
                    
                    cell_bbox = [
                        block.polygon.bbox[0] + (cur_col * col_width),
                        block.polygon.bbox[1] + (i * row_height),
                        block.polygon.bbox[0] + ((cur_col + colspan) * col_width),
                        block.polygon.bbox[1] + ((i + rowspan) * row_height),
                    ]
                    cell_polygon = PolygonBox.from_bbox(cell_bbox)
                    print(f"  -> Created new proportional cell: {cell_bbox}")

                cell_obj = TableCell(
                    text_lines=[cell_text],
                    row_id=i,
                    col_id=cur_col,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=cell.name == "th",
                    polygon=cell_polygon,
                    page_id=page.page_id,
                )
                cells.append(cell_obj)
                cur_col += colspan

        # Print final usage statistics
        if used_cells:
            print(f"Original cell usage statistics:")
            for cell_idx, count in used_cells.items():
                if cell_idx < len(original_children):
                    original_cell = original_children[cell_idx]
                    original_text = ' '.join(original_cell.text_lines) if original_cell.text_lines else ""
                    print(f"  Cell {cell_idx} at ({original_cell.row_id}, {original_cell.col_id}) with text '{original_text}': used {count} times")

        return cells
    
def find_closest_mapped_bbox(original_children: List[TableCell], new_index_row: int, new_index_col: int, new_text: str, used_cells: dict = None):
    """
    Find the closest matching cell from original_children with improved handling for:
    - Fuzzy text matching (spelling mistakes)
    - Cell merging (multiple new cells mapping to same original)
    - Row coverage tracking (accounting for rowspan/colspan)
    
    Args:
        original_children: List of original TableCell objects
        new_index_row: Row index of the new cell
        new_index_col: Column index of the new cell
        new_text: Text content of the new cell
        used_cells: Dict tracking usage count of original cells (using cell indices as keys)
        
    Returns:
        The polygon of the best matching original cell, or None if no good match is found
    """
    import difflib
    
    if not original_children:
        return None
    
    if used_cells is None:
        used_cells = {}
    
    # Build coverage map for original cells
    coverage_map = {}
    for i, cell in enumerate(original_children):
        for r in range(cell.row_id, cell.row_id + cell.rowspan):
            for c in range(cell.col_id, cell.col_id + cell.colspan):
                if (r, c) not in coverage_map:
                    coverage_map[(r, c)] = []
                coverage_map[(r, c)].append((i, cell))
    
    def fuzzy_text_similarity(text1: str, text2: str) -> float:
        """Calculate fuzzy text similarity using difflib.SequenceMatcher"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts - remove punctuation and extra spaces
        import re
        text1_norm = re.sub(r'[^\w\s]', '', text1.lower()).strip()
        text2_norm = re.sub(r'[^\w\s]', '', text2.lower()).strip()
        
        # Also check if text1 is contained in text2 or vice versa
        if text1_norm in text2_norm or text2_norm in text1_norm:
            return 0.9  # High similarity for substring matches
        
        # Use SequenceMatcher for fuzzy matching
        similarity = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
        return similarity
    
    def get_cell_text(cell: TableCell) -> str:
        """Get text content from a TableCell"""
        return ' '.join(cell.text_lines) if cell.text_lines else ""
    
    # First, calculate text similarities for all cells
    text_similarities = []
    for i, cell in enumerate(original_children):
        cell_text = get_cell_text(cell)
        text_sim = fuzzy_text_similarity(cell_text, new_text)
        text_similarities.append((i, cell, text_sim))
    
    # Strategy 1: High text similarity match (>0.7) - prioritize this over position
    high_text_matches = [(i, cell, sim) for i, cell, sim in text_similarities if sim > 0.7]
    if high_text_matches:
        # Sort by similarity, then by usage (prefer less used)
        cell_idx, best_match, _ = max(high_text_matches, key=lambda x: (x[2], -used_cells.get(x[0], 0)))
        used_cells[cell_idx] = used_cells.get(cell_idx, 0) + 1
        return best_match.polygon
    
    # Strategy 2: Exact position match (only if no high text similarity)
    exact_matches = [(i, cell) for i, cell in enumerate(original_children) if cell.row_id == new_index_row and cell.col_id == new_index_col]
    if exact_matches:
        cell_idx, best_match = exact_matches[0]
        used_cells[cell_idx] = used_cells.get(cell_idx, 0) + 1
        return best_match.polygon
    
    # Strategy 3: Cells that span this position
    spanning_matches = coverage_map.get((new_index_row, new_index_col), [])
    if spanning_matches:
        # If multiple cells span this position, prefer the one with best text match
        if len(spanning_matches) == 1:
            cell_idx, best_match = spanning_matches[0]
        else:
            text_scores = [(cell_idx, cell, fuzzy_text_similarity(get_cell_text(cell), new_text)) for cell_idx, cell in spanning_matches]
            cell_idx, best_match, _ = max(text_scores, key=lambda x: x[2])
        
        used_cells[cell_idx] = used_cells.get(cell_idx, 0) + 1
        return best_match.polygon
    
    # Strategy 4: Medium text similarity with position proximity
    candidates = []
    for i, cell in enumerate(original_children):
        cell_text = get_cell_text(cell)
        text_sim = fuzzy_text_similarity(cell_text, new_text)
        
        # Calculate position distance
        row_dist = abs(cell.row_id - new_index_row)
        col_dist = abs(cell.col_id - new_index_col)
        position_dist = row_dist + col_dist
        
        # Penalize heavily used cells but don't exclude them
        usage_penalty = used_cells.get(i, 0) * 0.1
        
        # Combined score: high text similarity, low position distance, low usage
        score = text_sim - (position_dist * 0.1) - usage_penalty
        
        candidates.append((i, cell, score, text_sim, position_dist))
    
    # Filter candidates with reasonable text similarity (>0.3) or very close position (<2)
    good_candidates = [
        (cell_idx, cell, score, text_sim, pos_dist) for cell_idx, cell, score, text_sim, pos_dist in candidates
        if text_sim > 0.3 or pos_dist < 2
    ]
    
    if good_candidates:
        # Sort by combined score
        cell_idx, best_match, _, _, _ = max(good_candidates, key=lambda x: x[2])
        used_cells[cell_idx] = used_cells.get(cell_idx, 0) + 1
        return best_match.polygon
    
    # Strategy 5: Closest available cell by position (fallback)
    if candidates:
        # Sort by position distance, then by usage (prefer less used cells)
        cell_idx, closest_match, _, _, _ = min(candidates, key=lambda x: (x[4], used_cells.get(x[0], 0)))
        used_cells[cell_idx] = used_cells.get(cell_idx, 0) + 1
        return closest_match.polygon
    
    return None


class TableSchema(BaseModel):
    comparison: str
    corrected_html: str
