import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Union

import PIL.Image
from pydantic import BaseModel, Field, create_model
from google import genai
from google.genai import types
from google.genai.errors import APIError
from io import BytesIO

def simple_gemini_call(
    prompt: str, 
    media: Union[PIL.Image.Image, List[PIL.Image.Image], str, Path, List[Union[PIL.Image.Image, str, Path]]], 
    response_schema: type[BaseModel] = None, 
    timeout: int = 60
):
    """
    Make a simplified call to Gemini API without additional validation
    
    Args:
        prompt: The text prompt to send to Gemini
        media: Can be:
            - A PIL Image
            - A list of PIL Images
            - A file path (str or Path) to an image or PDF
            - A list containing any combination of PIL Images and file paths
        response_schema: Pydantic schema for response structuring (optional)
        timeout: Timeout in seconds
        
    Returns:
        Dict containing the response from Gemini
    """
    # Initialize Gemini client
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        http_options={"timeout": timeout * 1000}
    )
    
    # Convert media to list if it's not already
    if not isinstance(media, list):
        media = [media]
    
    # Process each media item
    media_parts = []
    
    for item in media:
        if isinstance(item, PIL.Image.Image):
            # Handle PIL Image - convert to bytes
            image_bytes = BytesIO()
            item.save(image_bytes, format="WEBP")
            media_parts.append(
                types.Part.from_bytes(data=image_bytes.getvalue(), mime_type="image/webp")
            )
        elif isinstance(item, (str, Path)):
            # Handle file path - upload file
            file_path = str(item)
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Upload file to Gemini
            uploaded_file = client.files.upload(file=file_path)
            print(uploaded_file)
            media_parts.append(uploaded_file)
        else:
            raise ValueError(f"Unsupported media type: {type(item)}. Supported types: PIL.Image.Image, str, Path")
    
    try:
        # Prepare configuration
        config = {
            "temperature": 0,
            "response_mime_type": "application/json",
        }
        
        # Add response schema if provided
        if response_schema:
            config["response_schema"] = response_schema
        
        # Make the API call
        responses = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Updated to use the model from your example
            contents=media_parts + [prompt],  # Media first for better performance
            config=config,
        )
        
        # Extract the response
        output = responses.candidates[0].content.parts[0].text
        return json.loads(output)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {}

def to_snake_case(text):
    """
    Convert string to snake_case
    
    Args:
        text: String in any format (camelCase, space-separated, etc.)
        
    Returns:
        String in snake_case format
    """
    # First handle spaces, hyphens, etc.
    s1 = text.replace("-", " ").replace(".", " ")
    
    # Handle camelCase
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s1)
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    
    # Replace spaces with underscores and convert to lowercase
    return s1.replace(" ", "_").lower()

def convert_keys_to_snake_case(data):
    """
    Convert all dictionary keys to snake_case
    
    Args:
        data: Dictionary with keys in any format
        
    Returns:
        Dictionary with all keys in snake_case
    """
    if not isinstance(data, dict):
        return data
    
    return {to_snake_case(k): convert_keys_to_snake_case(v) if isinstance(v, dict) else v 
            for k, v in data.items()}

def detect_document_schema(file_path: Union[str, Path]) -> Dict:
    """
    Step 1: Detect the schema/structure from the document file
    
    Args:
        file_path: Path to the document file (image or PDF)
    
    Returns:
        Dict containing detected field names and their types as a JSON Schema
    """
    # Create the prompt for schema detection with snake_case instruction and JSON Schema output
    prompt = """
You are an AI assistant tasked with generating a JSON schema based on the document I will provide in the section "📜 Document for Analysis" below. Your primary goal is to meticulously analyze the content of this document—be it sample JSON, API specifications, textual descriptions of data, or any other format detailing data structures—and to derive a comprehensive, well-structured, and valid JSON schema. Keep the keys in Snake Case.
Your output should be

⚙️ Rules for JSON Schema Best Practices:
When constructing the JSON schema, adhere strictly to the following best practices:
1.  **Schema Core Attributes**:
    * `$schema`: Always include the `$schema` keyword, typically set to `"http://json-schema.org/draft-07/schema#"` or a newer draft if appropriate.
    * `title`: Provide a concise, human-readable title for the schema, especially for the root schema and any significant sub-schemas.
    * `description`: Write a clear, human-readable description explaining the purpose and structure of the data represented by the schema or field. This is crucial for maintainability and understanding.
    * `type`: Explicitly define the `type` for every field. Common types include:
        * `"object"`: For structured data with key-value pairs.
        * `"array"`: For ordered lists of items.
        * `"string"`: For textual data.
        * `"number"`: For floating-point or general numbers.
        * `"integer"`: For whole numbers.
        * `"boolean"`: For true or false values.
        * `"null"`: To explicitly allow a field to be null. Often used as part of a type array, e.g., `{"type": ["string", "null"]}`.

2.  **Object and Property Definitions**:
    * `properties`: For `type: "object"`, define all its properties within the `properties` keyword. (Ensure property names follow `snake_case` as per Naming Conventions).
    * `required`: List all mandatory properties of an object in a `required` array. Fields not listed are considered optional.
    * `additionalProperties`: By default, JSON Schema allows extra properties. If you want to restrict an object to only its defined properties, set `additionalProperties: false`. If some additional properties are allowed but should conform to a specific schema, define that schema under `additionalProperties`.

3.  **Array Definitions**:
    * `items`: For `type: "array"`, use the `items` keyword to define the schema for the elements within the array.
        * If all items are of the same type: `items: { "type": "string" }` or `items: { "$ref": "#/$defs/MyObjectDefinition" }`. (Note: `MyObjectDefinition` should be `PascalCase`).
        * If items can be of different types (tuple-like): `items: [{ "type": "string" }, { "type": "number" }]`.
    * `minItems` / `maxItems`: Specify the minimum and maximum number of items allowed in an array.
    * `uniqueItems`: Set to `true` if all items in the array must be unique.

4.  **Data Validation and Constraints**:
    * **Strings**:
        * `minLength` / `maxLength`: Define minimum/maximum string length.
        * `pattern`: Use a regular expression (ECMA 262) to validate string formats (e.g., email, UUID).
        * `format`: Utilize built-in formats like `"date-time"`, `"date"`, `"time"`, `"email"`, `"ipv4"`, `"uri"`, `"uuid"`.
    * **Numbers / Integers**:
        * `minimum` / `maximum`: Define inclusive numeric limits.
        * `exclusiveMinimum` / `exclusiveMaximum`: Define exclusive numeric limits.
        * `multipleOf`: Specify that the number must be a multiple of a given value.
    * `enum`: If a field (e.g. `order_status`) must be one of a predefined set of values, list them in an `enum` array. (Enum values should follow `UPPER_SNAKE_CASE` as per Naming Conventions).
    * `const`: If a field must always have a specific constant value.
    * `default`: Provide a default value for a field if one is commonly applicable. (Ensure default values match the field's type and format).

5.  **Reusability and Structure**:
    * `$defs` (or `definitions` for older drafts): Define reusable schema components (e.g., complex objects used in multiple places) under `$defs` at the root of your schema. Name these definitions using `PascalCase`.
    * `$ref`: Reference these reusable components using `"$ref": "#/$defs/MyReusableComponent"`. This promotes modularity and reduces redundancy.
    * Nesting: Represent hierarchical data naturally using nested objects and arrays.

6.  **Clarity and Examples**:
    * `examples`: Provide an array of one or more valid example values for each field. This greatly helps users understand the expected data. (Ensure example keys follow `snake_case`).
    * Comments (`$comment`): While not for validation, `$comment` can be used to add notes or explanations within the schema for developers.

🏷️ Naming Convention Best Practices:
Apply these naming conventions consistently. **For JSON property names (keys), `snake_case` is the strongly required standard.** For schema titles and definitions, use `PascalCase`. For enum values, prefer `UPPER_SNAKE_CASE`. Ensure consistency for all chosen conventions.

1.  **General Principles**:
    * Clarity and Descriptiveness: Names should be self-explanatory and clearly indicate the data they represent.
    * Consistency: The chosen naming style must be applied uniformly across the entire schema. **For property names, this means consistent `snake_case`.**
    * Avoid Ambiguity: Choose names that have a single, clear meaning.
    * Brevity (where appropriate): While descriptive, avoid overly verbose names. Find a balance.
    * Special Characters in Keys: Property names must be alphanumeric, with the underscore (`_`) used as a delimiter in `snake_case`. **`snake_case` uses underscores to separate words, enhancing readability for many developers.** Avoid spaces or hyphens. (If `camelCase` is used in exceptional, legacy scenarios, it avoids underscores).

2.  **Convention for Keys/Property Names: Use `snake_case`**
    * **Mandate `snake_case`**: This convention (e.g., `first_name`, `user_profile`, `item_count`, `order_is_enabled`) **must be used** for all JSON property names. It is a highly readable standard, common in many backend languages (e.g., Python, Ruby, PHP) and database schema designs, ensuring clarity and broad developer familiarity.
        * Example of a `snake_case` property definition:
            `"shipping_address": { "type": "string", "description": "The complete shipping address for the order." }`
        * Example of `snake_case` properties within an object schema:
            `"customer_details": { "type": "object", "properties": { "customer_id": { "type": "string" }, "contact_full_name": { "type": "string" }, "is_active_flag": { "type": "boolean", "examples": [true] } } }`
    * **Boolean Property Naming (using `snake_case`)**: For boolean fields, always use `snake_case` combined with prefixes like `is_`, `has_`, `can_`, `should_` (e.g., `is_active`, `has_subscription`, `can_purchase_item`, `should_send_notification`).
        * Example: `"is_opted_in_to_newsletter": { "type": "boolean", "default": false, "examples": [true, false] }`
    * **Exceptions for `camelCase` (Strictly Controlled and Avoided where Possible)**: `camelCase` (e.g., `firstName`, `userProfile`) is **not recommended** for new JSON schemas when `snake_case` is the chosen standard. Its use should be confined strictly to situations where interaction with an existing, unchangeable system (e.g., a JavaScript-heavy frontend that cannot easily adapt) or a non-negotiable legacy style guide explicitly mandates it for JSON interchange. If such an exception applies, `camelCase` must be used with absolute consistency. **When in doubt, or for any new development under a `snake_case` policy, always choose and enforce `snake_case`.**

3.  **Convention for Schema Titles and `$defs/definitions` Names: Use `PascalCase`**
    * Use `PascalCase` for naming schema titles and reusable definitions (e.g., `UserProfile`, `AddressDetails`, `OrderItemSchema`). This distinguishes them from `snake_case` property names.
        * Example in `$defs` (note `snake_case` for properties within):
            `"$defs": { "OrderLineItem": { "type": "object", "title": "Order Line Item Definition", "properties": { "product_id": {"type": "string", "examples": ["PROD_123"]}, "quantity_ordered": {"type": "integer", "examples": [1, 10]} } } }`
        * Example schema title: `"title": "ProductCatalogRecord"`

4.  **Convention for Enum Values: Use `UPPER_SNAKE_CASE`**
    * `UPPER_SNAKE_CASE` is strongly recommended for values within an `enum` list (e.g., `PENDING_APPROVAL`, `COMPLETED_SUCCESSFULLY`, `FAILED_VALIDATION`). This provides clear visual distinction for constant values.
        * Example using a `snake_case` property name:
            `"order_status": { "type": "string", "enum": ["ACTIVE", "INACTIVE", "PENDING_REVIEW", "ARCHIVED_DEPRECATED"], "examples": ["ACTIVE"] }`
    * While `PascalCase` (e.g., `PendingApproval`) or simple string literals (e.g., "active", "in-progress") might be encountered in existing systems, for new schemas, `UPPER_SNAKE_CASE` should be the standard.

**Concluding Reinforcement of Naming Conventions:**
Remember, the primary and strongly enforced principle for JSON property naming is **`snake_case`**. This ensures your schemas align with readability preferences common in many backend and data-centric environments. Complement this with `PascalCase` for schema titles and definitions, and `UPPER_SNAKE_CASE` for enumeration values to maintain a clear, consistent, and maintainable naming strategy across your entire JSON schema.

Remember: Your final output should be ONLY the JSON schema code block. Analyze the provided document thoroughly and apply all above guidelines to generate the most accurate and well-formed schema.
    """
    
    # Use simple Gemini call without a predefined schema, passing the file path directly
    schema_result = simple_gemini_call(
        prompt=prompt,
        media=file_path,
        timeout=60
    )
    
    print("Detected schema (JSON Schema):")
    print(json.dumps(schema_result, indent=2))
    
    return schema_result

def create_dynamic_model(schema_fields: Dict[str, str]) -> type[BaseModel]:
    """
    Create a dynamic Pydantic model based on detected schema fields
    
    Args:
        schema_fields: Dictionary of field names and their types
    
    Returns:
        Dynamically created Pydantic model class
    """
    field_definitions = {}
    
    # Convert the schema fields to Pydantic field definitions
    for field_name, field_type in schema_fields.items():
        # For simplicity, treating all fields as strings
        # In a production system, you might want to handle different types
        field_definitions[field_name] = (str, ...)
    
    # Create the dynamic model
    DynamicModel = create_model(
        'DynamicDocumentModel',
        **field_definitions
    )
    
    return DynamicModel

def extract_document_data(file_path: Union[str, Path]) -> Dict:
    """
    Two-step extraction process:
    1. Detect schema from document
    2. Extract data based on that schema
    
    Args:
        file_path: Path to the document file (image or PDF)
    
    Returns:
        Dict containing extracted document data
    """
    # Step 1: Detect the schema
    schema_fields = detect_document_schema(file_path)
    
    # Step 2: Create a dynamic model based on the detected schema
    DynamicModel = create_dynamic_model(schema_fields)
    
    # Create the prompt for data extraction
    prompt = """
    Extract all the information from this document according to the detected schema.
    Return the data in a structured JSON format matching all the fields identified.
    Ensure all values are accurate and complete.
    """
    
    # Use the simple Gemini call with the dynamic model
    result = simple_gemini_call(
        prompt=prompt,
        media=file_path,
        response_schema=DynamicModel,
        timeout=60
    )
    
    return result

def extract_document_data_with_schema(file_path: Union[str, Path], json_schema: dict) -> dict:
    """
    Extract data from the document file using the provided JSON Schema.
    Args:
        file_path: Path to the document file (image or PDF)
        json_schema: JSON Schema dict (must have 'properties')
    Returns:
        Dict containing extracted document data
    """
    # Build a dynamic Pydantic model from the JSON Schema 'properties'
    properties = json_schema.get("properties", {})
    field_definitions = {}
    for field_name, field_schema in properties.items():
        # Map JSON Schema types to Python types (default to str)
        t = field_schema.get("type", "string")
        py_type = str
        if t == "number":
            py_type = float
        elif t == "integer":
            py_type = int
        elif t == "boolean":
            py_type = bool
        field_definitions[field_name] = (py_type, ...)
    DynamicModel = create_model('DynamicDocumentModel', **field_definitions)
    
    # Create the prompt for data extraction
    prompt = """
    Extract all the information from this document according to the provided schema.
    Return the data in a structured JSON format matching all the fields in the schema.
    Ensure all values are accurate and complete.
    
    JSON Schema:
    {json_schema}
    """.format(json_schema=json.dumps(json_schema, indent=2))
    
    print(prompt)
    
    # Use the simple Gemini call with the dynamic model
    result = simple_gemini_call(
        prompt=prompt,
        media=file_path,
        # response_schema=DynamicModel,
        timeout=60
    )
    return result

if __name__ == "__main__":
    # Path to the image
    image_path = Path(__file__).parent / "tests" / "Documents" / "Set 1" / "Purchase Contract.png"
    
    # Extract information using the two-step process
    document_data = extract_document_data(str(image_path))
    
    # Print the extracted information
    print("\nExtracted Data:")
    print(json.dumps(document_data, indent=2))
