from marker.processors.llm import BaseLLMProcessor
import os
import json
import base64
import logging
import time
import math
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union
from marker.schema.document import Document
from marker.schema import BlockTypes
from marker.processors.llm import PromptData, BaseLLMSimpleBlockProcessor, BlockData
from pydantic import BaseModel

from typing import Annotated, List

logger = logging.getLogger(__name__)

class LLMDubiousTextProcessor(BaseLLMSimpleBlockProcessor):
    """
    Processor that uses LLM to analyze document content using parallel processing.
    """
    
    block_types = (BlockTypes.Text, BlockTypes.Table, )
    
    min_text_height: Annotated[
        float,
        "The minimum ratio between text height and page height to consider for processing.",
     ] = 0.06
    
    image_expansion_ratio: Annotated[
        float,
        "The ratio to expand the image by when cropping.",
    ] = 0.20
    
    redo_text: Annotated[
        bool,
        "Whether to redo text blocks.",
    ] = False
    
    check_dubious_text_prompt: Annotated[
        str,
        "The prompt to use for checking dubious text.",
        "Default is a string containing the Gemini prompt."
    ] = r"""You're an expert in checking if text is dubious.  You'll be given a block of text from a document.  Your job is to determine if the text is dubious.
    Dubious text is something that user modified post writing it. There are two types of dubious text:
    1. Text that is strikethrough.
    2. Text that is overwritten.
    
    Some guidelines:
    - Because of the criss crossing, the input html provided may have wrongly identified text. Fix that as well.
    - Output valid html.
    - Use <strike> for strikethrough text and <over> for overwritten text.
    - A text block may have multiple words that are strikethrough or overwritten. Only do this for words that are clearly strikethrough or overwritten.
    - Make sure to include both the original text and the modified text in the output.
    
    **Example:**
    Input: 
    An image having handwritten `jumps` being strikethrough and `lazy` being overwritten to `crazy`.
    +
    ```html
    <table><tbody><tr><th>3</th><th>Monoethylamine</th><th>42 L</th><th></th><th>M002</th></tr><tr><td>4</td><td>Sodium Hydorxide (NaOH)</td><td></td><th></th><td>M004</td></tr><tr><td>5</td><td>Hydrochloric Acid (HCI)</td><td>10 %</td><th></th><td>M005</td></tr><tr><td>6</td><td>L-Glutamine Solution</td><td></td><th></th><td>M020 M015</td></tr><tr><td>7</td><td>Sodium Bicarbonate</td><td></td><th></th><td>M057 M020</td></tr><tr><td>8</td><td>Phosphate Buffered Saline</td><td></td><th></th><td>M000 M05</td></tr><tr><td>9</td><td>Tween 20</td><td></td><th></th><td>M006</td></tr><tr><td>10</td><td>HEPES Buffer</td><td></td><th></th><td>M024</td></tr><tr><td>11</td><td>TheraPro CHO feed</td><td>35.37 kg</td><td>Sartorius</td><td>SH31113.01 M045</td></tr></tbody></table>
    ```
    Output:
    ```html
    <table>
        <tbody>
            <tr><th>3</th><th>Monoethylamine</th><th>42 L</th><th></th><th>M002</th></tr>
            <tr><td>4</td><td>Sodium Hydorxide (NaOH)</td><td></td><th></th><td>M004</td></tr>
            <tr><td>5</td><td>Hydrochloric Acid (HCI)</td><td>10 %</td><th></th><td>M005</td></tr>
            <tr><td>6</td><td>L-Glutamine Solution</td><td></td><th></th><td> <strike>M020</strike> <over>M015</over></td></tr>
            <tr><td>7</td><td>Sodium Bicarbonate</td><td></td><th></th><td> <strike>M057</strike> <over>M020</over></td></tr>
            <tr><td>8</td><td>Phosphate Buffered Saline</td><td></td><th></th><td>M000 M05</td></tr>
            <tr><td>9</td><td>Tween 20</td><td></td><th></th><td>M006</td></tr>
            <tr><td>10</td><td>HEPES Buffer</td><td></td><th></th><td>M024</td></tr>
            <tr><td>11</td><td>TheraPro CHO feed</td><td>35.37 kg</td><td>Sartorius</td><td>SH31113.01 M045</td></tr>
        </tbody>
    </table>
    ```
    
**Input:**
```html
{text}
```    
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        """
        Inference the dubious text in the document.
        """
        return super().inference_blocks(document)
    
    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            text = block.html if block.html else block.raw_text(document)
            prompt = self.check_dubious_text_prompt.replace("{equation}", text)
            image = self.extract_image(document, block)

            prompt_data.append({
                "prompt": prompt,
                "image": image,
                "block": block,
                "schema": DubiousTextSchema,
                "page": block_data["page"]
            })

        return prompt_data
    
    def rewrite_block(self, response: dict, prompt_data: PromptData, document: Document):
        block = prompt_data["block"]
        text = block.html if block.html else block.raw_text(document)

        if not response or "is_dubious" not in response:
            block.update_metadata(llm_error_count=1)
            return
        
        html_text = response["corrected_text"]
        is_dubious = response["is_dubious"]
        is_strikethrough = response.get("is_strikethrough", False)

        if is_dubious:
            parsed_cell_text = html_text
            orig_cell_text = text
            # Potentially a partial response
            print("Parsed Table size", len(parsed_cell_text))
            if len(parsed_cell_text) < len(orig_cell_text) * 0.5:
                print("LLM Error - Partial Response")
                block.update_metadata(llm_error_count=1)
                return
            
            print("Updated Dubious Cells")
            dubious_reason = response.get("dubious_reason")
            block.html = html_text
            block.is_dubious = is_dubious
            block.dubious_reason = dubious_reason
        elif is_strikethrough:
            print("No Coorections Found - is_strikethrough")
            return
        else:
            print("No Coorections Found - is_dubious False", is_dubious)
            return


class DubiousTextSchema(BaseModel):
    analysis: str
    corrected_text: str
    is_dubious: bool
    dubious_reason: str | None = None
    is_strikethrough: bool = False
