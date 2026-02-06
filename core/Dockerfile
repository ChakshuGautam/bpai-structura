FROM python:3.10-slim

WORKDIR /app

# Copy only dependency files first for better caching
COPY core/requirements.txt ./requirements.txt
COPY core/marker/pyproject.toml /app/core/marker/pyproject.toml

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install modal
RUN pip install poetry
RUN sed -i '/^torch =/d' /app/core/marker/pyproject.toml
RUN rm -f /app/core/marker/poetry.lock

# Now copy the rest of the source code (including README.md)
COPY core/ core/

WORKDIR /app/core/marker
RUN pip install -e .
RUN pip show marker || pip list | grep marker
RUN python -c "import marker; print(marker.__file__)"

WORKDIR /app

# Make core importable as a top-level package
ENV PYTHONPATH=/app

EXPOSE 8000

RUN chmod +x /app/core/entrypoint.sh

ENTRYPOINT ["/app/core/entrypoint.sh"] 