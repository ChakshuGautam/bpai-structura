#!/bin/bash
set -e

# Set modal token using environment variables
modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET

# Start the server
exec uvicorn core.server:app --host 0.0.0.0 --port 8000 