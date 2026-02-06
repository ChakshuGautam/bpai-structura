# Structura API Documentation

## Table of contents

- General Info
- Authentication
- Limits
  - Rate Limits
  - File size limits
- Callbacks
- API Endpoints
  - Document Conversion
    - Response fields
    - Supported file types
    - Troubleshooting
  - Table Recognition
    - Response fields
    - Supported file types
  - OCR
    - Response fields
    - Supported file types
  - Layout
    - Response fields
    - Supported file types

## General Info

These hosted APIs leverage advanced AI models to do OCR, convert various document formats to markdown, and recognize tables.

**Note:** These APIs are currently under active development and iteration. Endpoints, parameters, and response formats may change without prior notice.

## Authentication

You authenticate by setting the `X-Api-Key` header. You can find your API keys [here](link-to-api-keys-page). Billing limits can be set per-key to avoid high bills.

## Limits

### Rate Limits

The request limit for all endpoints is 200 per 60 seconds. You also cannot have more than 200 concurrent requests. You'll get a `429` error if you exceed the rate limit.

Reach out to [contact@theflywheel.in](mailto:contact@theflywheel.in) if you need higher limits.

### File size limits

The file size limit for all endpoints is 200MB currently. If you need to submit larger files, please reach out to [contact@theflywheel.in](mailto:contact@theflywheel.in). One solution is also to slice the file into chunks that are under 200MB in size.

## Callbacks

You will normally need to poll to get API results. If you don't want to poll, you can specify a URL that will be hit when inference is complete. Specify the webhook URL in the settings panel on the dashboard.

The callback will pass this data to your webhook URL:

- `request_id`: lookup key of the original request
- `webhook_secret`: a webhook secret you can define to reject other messages
- `request_check_url`: the url you will need to hit to get the full results

## API Endpoints

All endpoints will return immediately, and continue processing in the background.

### Document Conversion

The document conversion endpoint converts PDFs, spreadsheets, word documents, epub, HTML, and powerpoints to markdown. It is available at `/api/v1/convert`.

**Example Request (Python):**

```python
import requests

url = "https://structura.try-veil.com/api/v1/convert" # Updated base URL and endpoint

form_data = {
    'file': ('test.pdf', open('~/pdfs/test.pdf', 'rb'), 'application/pdf'),
    'langs': (None, "English"),
    "force_ocr": (None, False),
    "paginate": (None, False),
    'output_format': (None, 'markdown'),
    "use_llm": (None, False),
    "strip_existing_ocr": (None, False),
    "disable_image_extraction": (None, False)
}

headers = {"X-Api-Key": "YOUR_API_KEY"}

response = requests.post(url, files=form_data, headers=headers)
data = response.json()
```

As you can see, everything is a form parameter. This is because we're uploading a file, so the request body has to be multipart/form-data.

**Parameters:**

- `file`: The input file.
- `langs`: An optional, comma-separated list of languages in the file (this is used if OCR is needed). The language names and codes are from [here](link-to-language-codes).
- `output_format`: One of `json`, `html`, or `markdown`.
- `force_ocr`: (False by default) Will force OCR on every page (ignore the text in the PDF). This is slower, but can be useful for PDFs with known bad text.
- `paginate`: (False by default) Adds delimiters to the output pages. See the API reference for details.
- `use_llm`: (False by default) Setting this to True will use an LLM to enhance accuracy of forms, tables, inline math, and layout. It can be much more accurate, but carries a small hallucination risk. Setting `use_llm` to True will make responses slower.
- `strip_existing_ocr`: (False by default) Setting to True will remove all existing OCR text from the file and redo OCR. This is useful if you know OCR text was added to the PDF by a low-quality OCR tool.
- `disable_image_extraction`: (False by default) Setting to True will disable extraction of images. If `use_llm` is set to True, this will also turn images into text descriptions.
- `max_pages`: From the start of the file, specifies the maximum number of pages to inference.

You can see a full list of parameters and descriptions in the API reference.

**Initial Response:**
The request will return the following:

```json
{
    "success": true,
    "error": null,
    "request_id": "PpK1oM-HB4RgrhsQhVb2uQ",
    "request_check_url": "https://structura.try-veil.com/api/v1/convert/PpK1oM-HB4RgrhsQhVb2uQ" # Updated base URL and endpoint
}
```

**Polling:**
You will then need to poll `request_check_url`, like this:

```python
import time
import requests # Make sure requests is imported

# Assuming 'data' holds the initial response and 'headers' are defined
check_url = data["request_check_url"]
max_polls = 300

for i in range(max_polls):
    time.sleep(2)
    response = requests.get(check_url, headers=headers) # Don't forget to send the auth headers
    poll_data = response.json() # Use a different variable name to avoid confusion

    if poll_data.get("status") == "complete": # Use .get() for safety
        break
    elif poll_data.get("status") == "error": # Handle potential errors during processing
        print(f"Error during processing: {poll_data.get('error')}")
        break
# After the loop, 'poll_data' will contain the final response if status is 'complete'
```

You can customize the max number of polls and the check interval to your liking. Eventually, the `status` field will be set to `complete`.

**Final Response:**
You will get an object that looks like this:

```json
{
    "output_format": "markdown",
    "markdown": "...",
    "status": "complete",
    "success": true,
    "images": {...},
    "metadata": {...},
    "error": "",
    "page_count": 5
}
```

If `success` is `False`, you will get an error code along with the response.

_Note: All response data will be deleted from Structura servers an hour after the processing is complete, so make sure to get your results by then._

#### Response fields

- `output_format`: The requested output format, `json`, `html`, or `markdown`.
- `markdown` | `json` | `html`: The output from the file. It will be named according to the `output_format`. You can find more details on the json format [here](link-to-json-format).
- `status`: Indicates the status of the request (`complete`, or `processing`).
- `success`: Indicates if the request completed successfully. `True` or `False`.
- `images`: Dictionary of image filenames (keys) and base64 encoded images (values). Each value can be decoded with `base64.b64decode(value)`. Then it can be saved to the filename (key).
- `metadata`: Metadata about the markdown conversion. (Note: documentation uses `meta` in some places but example shows `metadata`. Assuming `metadata` is correct).
- `error`: If there was an error, this contains the error message.
- `page_count`: Number of pages that were converted.

#### Supported file types

The Document Conversion endpoint supports the following extensions and mime types:

- PDF - `pdf`/`application/pdf`
- Spreadsheet - `xls`/`application/vnd.ms-excel`, `xlsx`/`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`, `ods`/`application/vnd.oasis.opendocument.spreadsheet`
- Word document - `doc`/`application/msword`, `docx`/`application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `odt`/`application/vnd.oasis.opendocument.text`
- Powerpoint - `ppt`/`application/vnd.ms-powerpoint`, `pptx`/`application/vnd.openxmlformats-officedocument.presentationml.presentation`, `odp`/`application/vnd.oasis.opendocument.presentation`
- HTML - `html`/`text/html`
- Epub - `epub`/`application/epub+zip`
- Images - `png`/`image/png`, `jpeg`/`image/jpeg`, `webp`/`image/webp`, `gif`/`image/gif`, `tiff`/`image/tiff`, `jpg`/`image/jpg`

You can automatically find the mimetype in python by installing `filetype`, then using `filetype.guess(FILEPATH).mime`.

#### Troubleshooting

If you get bad output, setting `force_ocr` to `True` is a good first step. A lot of PDFs have bad text inside. Structura attempts to auto-detect this and run OCR, but the auto-detection is not 100% accurate. Making sure `langs` is set properly is a good second step.

### Table Recognition

The table recognition endpoint at `/api/v1/table_rec` will detect tables, then identify the structure and format the tables properly.

**Example Request (Python):**

```python
import requests

url = "https://structura.try-veil.com/api/v1/table_rec" # Updated base URL

form_data = {
    'file': ('test.png', open('~/images/test.png', 'rb'), 'image/png'),
}

headers = {"X-Api-Key": "YOUR_API_KEY"}

response = requests.post(url, files=form_data, headers=headers)
data = response.json()
```

The api accepts files of type `application/pdf`, `image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/tiff`, and `image/jpg`.

**Optional Parameters:**

- `max_pages`: Lets you specify the maximum number of pages of a pdf to make predictions for.
- `skip_table_detection`: Doesn't re-detect tables if your pages are already cropped.
- `detect_cell_boxes`: Will re-detect all cell bounding boxes vs using the text in the PDF.
- `output_format`: (`markdown` by default) The format of the output, one of `markdown`, `html`, `json`.
- `paginate`: (`False` by default) Determines whether to paginate markdown output.
- `use_llm`: (`False` by default) Optionally uses an LLM to merge tables and improve accuracy. This can be much more accurate, but has a small hallucination risk. It also doubles the per-page cost. Setting `use_llm` to `True` will make responses slower.

**Initial Response:**

```json
{
    "success": true,
    "error": null,
    "request_id": "PpK1oM-HB4RgrhsQhVb2uQ",
    "request_check_url": "https://structura.try-veil.com/api/v1/table_rec/PpK1oM-HB4RgrhsQhVb2uQ" # Updated base URL
}
```

**Polling:**
(Use the same polling logic as described in the Document Conversion section)

**Final Response:**

```json
{
    "output_format": "markdown",
    "markdown": "...",
    "status": "complete",
    "success": true,
    "metadata": {...},
    "error": "",
    "page_count": 5
}
```

If `success` is `False`, you will get an error code along with the response.

_Note: All response data will be deleted from Structura servers an hour after the processing is complete, so make sure to get your results by then._

#### Response fields

- `output_format`: The requested output format, `json`, `html`, or `markdown`.
- `markdown` | `json` | `html`: The output from the file. It will be named according to the `output_format`. You can find more details on the json format [here](link-to-json-format).
- `status`: Indicates the status of the request (`complete`, or `processing`).
- `success`: Indicates if the request completed successfully. `True` or `False`.
- `metadata`: Metadata about the markdown conversion. (Note: documentation uses `meta` in some places but example shows `metadata`. Assuming `metadata` is correct).
- `error`: If there was an error, this contains the error message.
- `page_count`: Number of pages that were converted.

#### Supported file types

Table recognition supports the same extensions and mime types as the Document Conversion endpoint.

### OCR

The OCR endpoint at `/api/v1/ocr` will OCR a given pdf, word document, powerpoint, or image.

**Example Request (Python):**

```python
import requests

url = "https://structura.try-veil.com/api/v1/ocr" # Updated base URL

form_data = {
    'file': ('test.png', open('~/images/test.png', 'rb'), 'image/png'),
    'langs': (None, "English") # Optional
}

headers = {"X-Api-Key": "YOUR_API_KEY"}

response = requests.post(url, files=form_data, headers=headers)
data = response.json()
```

The `langs` field is optional, to give language hints to the OCR model. The language names and codes are from [here](link-to-language-codes). You can specify up to 4 languages.

The api accepts files of type `application/pdf`, `image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/tiff`, and `image/jpg`.

**Optional Parameters:**

- `max_pages`: Lets you specify the maximum number of pages of a pdf to make predictions for.

**Initial Response:**

```json
{
    "success": true,
    "error": null,
    "request_id": "PpK1oM-HB4RgrhsQhVb2uQ",
    "request_check_url": "https://structura.try-veil.com/api/v1/ocr/PpK1oM-HB4RgrhsQhVb2uQ" # Updated base URL
}
```

**Polling:**
(Use the same polling logic as described in the Document Conversion section)

**Final Response:**

```json
{
    "status": "complete",
    "pages": [
        {
            "text_lines": [{
                "polygon": [[267.0, 139.0], [525.0, 139.0], [525.0, 159.0], [267.0, 159.0]],
                "confidence": 0.99951171875,
                "text": "Subspace Adversarial Training",
                "bbox": [267.0, 139.0, 525.0, 159.0]
            }, ...],
            "languages": ["en"],
            "image_bbox": [0.0, 0.0, 816.0, 1056.0],
            "page": 12
        }
    ],
    "success": true,
    "error": "",
    "page_count": 5
}
```

If `success` is `False`, you will get an error code along with the response.

_Note: All response data will be deleted from Structura servers an hour after the processing is complete, so make sure to get your results by then._

#### Response fields

- `status`: Indicates the status of the request (`complete`, or `processing`).
- `success`: Indicates if the request completed successfully. `True` or `False`.
- `error`: If there was an error, this is the error message.
- `page_count`: Number of pages we ran ocr on.
- `pages`: A list containing one dictionary per input page. The fields are:
  - `text_lines`: The detected text and bounding boxes for each line.
    - `text`: The text in the line.
    - `confidence`: The confidence of the model in the detected text (0-1).
    - `polygon`: The polygon for the text line in `(x1, y1), (x2, y2), (x3, y3), (x4, y4)` format. The points are in clockwise order from the top left.
    - `bbox`: The axis-aligned rectangle for the text line in `(x1, y1, x2, y2)` format. `(x1, y1)` is the top left corner, and `(x2, y2)` is the bottom right corner.
  - `languages`: The languages specified for the page.
  - `page`: The page number in the file.
  - `image_bbox`: The bbox for the image in `(x1, y1, x2, y2)` format. `(x1, y1)` is the top left corner, and `(x2, y2)` is the bottom right corner. All line bboxes will be contained within this bbox.

#### Supported file types

OCR supports the same extensions and mime types as the Document Conversion endpoint.

### Layout

The layout endpoint at `/api/v1/layout` will detect layout bboxes in a given pdf, word document, powerpoint, or image. The possible labels for the layout bboxes are: `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Figure`, `Section-header`, `Table`, `Text`, `Title`. The layout boxes are then labeled with a `position` field indicating their reading order, and sorted.

**Example Request (Python):**

```python
import requests

url = "https://structura.try-veil.com/api/v1/layout" # Updated base URL

form_data = {
    'file': ('test.png', open('~/images/test.png', 'rb'), 'image/png'),
}

headers = {"X-Api-Key": "YOUR_API_KEY"}

response = requests.post(url, files=form_data, headers=headers)
data = response.json()
```

The api accepts files of type `application/pdf`, `image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/tiff`, and `image/jpg`.

**Optional Parameters:**

- `max_pages`: Lets you specify the maximum number of pages of a pdf to make predictions for.

**Initial Response:**

```json
{
    "success": true,
    "error": null,
    "request_id": "PpK1oM-HB4RgrhsQhVb2uQ",
    "request_check_url": "https://structura.try-veil.com/api/v1/layout/PpK1oM-HB4RgrhsQhVb2uQ" # Updated base URL
}
```

**Polling:**
(Use the same polling logic as described in the Document Conversion section)

**Final Response:**

```json
{
  "status": "complete",
  "pages": [
    {
      "bboxes": [
        {
          "bbox": [0.0, 0.0, 1334.0, 1625.0],
          "position": 0,
          "label": "Table",
          "polygon": [
            [0, 0],
            [1334, 0],
            [1334, 1625],
            [0, 1625]
          ]
        }
      ],
      "image_bbox": [0.0, 0.0, 1336.0, 1626.0],
      "page": 1
    }
  ],
  "success": true,
  "error": "",
  "page_count": 5
}
```

If `success` is `False`, you will get an error code along with the response.

_Note: All response data will be deleted from Structura servers an hour after the processing is complete, so make sure to get your results by then._

#### Response fields

- `status`: Indicates the status of the request (`complete`, or `processing`).
- `success`: Indicates if the request completed successfully. `True` or `False`.
- `error`: If there was an error, this is the error message.
- `page_count`: Number of pages we ran layout on.
- `pages`: A list containing one dictionary per input page. The fields are:
  - `bboxes`: Detected layout bounding boxes.
    - `bbox`: The axis-aligned rectangle for the text line in `(x1, y1, x2, y2)` format. `(x1, y1)` is the top left corner, and `(x2, y2)` is the bottom right corner.
    - `polygon`: The polygon for the text line in `(x1, y1), (x2, y2), (x3, y3), (x4, y4)` format. The points are in clockwise order from the top left.
    - `label`: The label for the bbox. One of `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Figure`, `Section-header`, `Table`, `Text`, `Title`.
    - `position`: The reading order of this bbox within the page.
  - `page`: The page number in the input file.
  - `image_bbox`: The bbox for the page image in `(x1, y1, x2, y2)` format. `(x1, y1)` is the top left corner, and `(x2, y2)` is the bottom right corner. All layout bboxes will be contained within this bbox.

#### Supported file types

Layout supports the same extensions and mime types as the Document Conversion endpoint.
