#!/bin/bash
# Databricks cluster init script for Structura parse_pdf.
# Uploaded to dbfs:/structura/init_script.sh
# Installs system dependencies and pre-downloads Marker models.

set -euo pipefail

echo "=== Structura init script starting ==="

# Install poppler-utils (required by pdf2image for PDF -> image conversion)
apt-get update -qq && apt-get install -y --no-install-recommends poppler-utils

# Pre-download Marker models to a persistent DBFS cache path
# This avoids re-downloading on every job run
CACHE_DIR="/dbfs/structura/model_cache"
export TORCH_HOME="${CACHE_DIR}/torch"
export HF_HOME="${CACHE_DIR}/huggingface"

mkdir -p "${TORCH_HOME}" "${HF_HOME}"

echo "Pre-downloading Marker models to ${CACHE_DIR}..."
python -c "
import os
os.environ['TORCH_HOME'] = '${TORCH_HOME}'
os.environ['HF_HOME'] = '${HF_HOME}'
from marker.models import create_model_dict
create_model_dict()
print('Models downloaded successfully')
"

echo "=== Structura init script complete ==="
