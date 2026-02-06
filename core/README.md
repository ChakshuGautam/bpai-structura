# Marker API FastAPI Server

This project provides a FastAPI server that wraps the Marker API for PDF-to-JSON conversion.

## Features

- **POST /api/v1/convert**: Upload a PDF and receive a task ID for asynchronous processing.
- **GET /api/v1/convert/{task_id}**: Poll for the result of a previously submitted PDF using the task ID.

## Setup

### Local (Python)

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**

   - `STRUCTURA_API_KEY`: API key for authenticating requests to this server.
   - `MARKER_API_KEY`: API key for authenticating requests to the Marker API.

3. **Run the server:**
   ```bash
   uvicorn core.server:app --reload
   ```

### Docker

1. **Build the Docker image (from project root):**
   ```bash
   docker build -f core/Dockerfile -t marker-api-server .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e STRUCTURA_API_KEY=changeme \
     -e MARKER_API_KEY=changeme \
     -e STRUCTURA_DOMAIN=http://localhost:8000 \
     marker-api-server
   ```

### Docker Compose

1. **Start the service:**
   ```bash
   docker-compose up --build
   ```
2. **Override environment variables:**
   Edit `.env` or use `docker-compose run -e VAR=VALUE ...`.

## API Usage

### 1. Submit a PDF for conversion

- **Endpoint:** `POST /api/v1/convert`
- **Headers:**
  - `X-Api-Key: <your STRUCTURA_API_KEY>`
- **Body:**
  - `file`: PDF file (multipart/form-data)
- **Response:**
  ```json
  {
    "success": true,
    "request_id": "...",
    "request_check_url": "..."
  }
  ```

### 2. Poll for conversion result

- **Endpoint:** `GET /api/v1/convert/{task_id}`
- **Headers:**
  - `X-Api-Key: <your STRUCTURA_API_KEY>`
- **Response:**
  - Returns the Marker API output or status.

## Project Structure

- `core/server.py`: FastAPI server with endpoints.
- `core/marker.py`: Marker API integration logic.
- `core/requirements.txt`: Python dependencies.
- `core/README.md`: This file.
- `core/Dockerfile`, `docker-compose.yml`: Containerization files.

## Notes

- The server uses direct polling to the Marker API for results.
- Only PDF files are supported.
