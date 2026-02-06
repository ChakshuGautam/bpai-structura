FROM python:3.10-slim

WORKDIR /app

# Install poppler for pdf2image (used by v3 endpoint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better caching
COPY core/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install modal
RUN pip install httpx

# Copy the core source code
COPY core/ core/

# Copy overrides on top of core (replaces supabase with psycopg2)
COPY bpai/overrides/db/ core/db/
COPY bpai/overrides/server.py core/server.py
COPY bpai/overrides/api/routes/convert.py core/api/routes/convert.py

# Make core importable as a top-level package
ENV PYTHONPATH=/app

EXPOSE 8000

COPY core/entrypoint.sh /app/core/entrypoint.sh
RUN chmod +x /app/core/entrypoint.sh

ENTRYPOINT ["/app/core/entrypoint.sh"]
