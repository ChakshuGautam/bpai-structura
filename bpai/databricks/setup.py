"""
One-time Databricks setup script for Structura parse_pdf.

Run this once to:
1. Create the 'structura' secret scope with all required secrets
2. Upload parse_pdf.py and init_script.sh to Workspace
3. Create the Databricks job with GPU cluster config
4. Print the DATABRICKS_JOB_ID to add to .env.prod

Usage:
    export DATABRICKS_HOST=https://adb-XXXX.XX.azuredatabricks.net
    export DATABRICKS_TOKEN=dapi...
    python bpai/databricks/setup.py
"""
import os
import sys
import json
import base64
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.")
    sys.exit(1)

HOST = DATABRICKS_HOST.rstrip("/")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}


def api(method: str, path: str, **kwargs) -> dict:
    """Make a Databricks REST API call."""
    url = f"{HOST}{path}"
    resp = requests.request(method, url, headers=HEADERS, **kwargs)
    if resp.status_code >= 400:
        logger.error(f"{method} {path} -> {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json() if resp.text else {}


# ──────────────────────────────────────────────
# 1. Create secret scope and populate secrets
# ──────────────────────────────────────────────

def setup_secrets():
    """Create the 'structura' secret scope and add all required secrets."""
    scope = "structura"

    # Create scope (ignore error if it already exists)
    try:
        api("POST", "/api/2.0/secrets/scopes/create", json={"scope": scope})
        logger.info(f"Created secret scope: {scope}")
    except requests.HTTPError as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e.response.text):
            logger.info(f"Secret scope '{scope}' already exists")
        else:
            raise

    # Secrets to configure — values read from current environment
    secret_keys = [
        "GOOGLE_API_KEY",
        "DO_ENDPOINT_URL",
        "DO_ACCESS_KEY_ID",
        "DO_SECRET_ACCESS_KEY",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_USERNAME",
        "REDIS_PASSWORD",
    ]

    for key in secret_keys:
        value = os.environ.get(key, "")
        if not value:
            logger.warning(f"  {key} is empty in current environment, setting empty secret")
        api("POST", "/api/2.0/secrets/put", json={
            "scope": scope,
            "key": key,
            "string_value": value,
        })
        logger.info(f"  Set secret: {scope}/{key}")

    logger.info("Secrets configured.")


# ──────────────────────────────────────────────
# 2. Upload files to Workspace
# ──────────────────────────────────────────────

WORKSPACE_DIR = "/Workspace/structura"


def ensure_workspace_dir():
    """Create the /Workspace/structura directory if it doesn't exist."""
    try:
        api("POST", "/api/2.0/workspace/mkdirs", json={"path": WORKSPACE_DIR})
        logger.info(f"Ensured workspace directory: {WORKSPACE_DIR}")
    except requests.HTTPError:
        # Directory may already exist
        pass


def upload_to_workspace(local_path: str, workspace_path: str):
    """Upload a local file to Databricks Workspace filesystem."""
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    # Workspace import API — language=AUTO, format=SOURCE, overwrite=True
    api("POST", "/api/2.0/workspace/import", json={
        "path": workspace_path,
        "content": content,
        "format": "AUTO",
        "overwrite": True,
    })
    logger.info(f"Uploaded {local_path} -> {workspace_path}")


def upload_artifacts():
    """Upload parse_pdf.py and init_script.sh to Workspace."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ensure_workspace_dir()

    upload_to_workspace(
        os.path.join(script_dir, "parse_pdf.py"),
        f"{WORKSPACE_DIR}/parse_pdf.py",
    )
    upload_to_workspace(
        os.path.join(script_dir, "init_script.sh"),
        f"{WORKSPACE_DIR}/init_script.sh",
    )
    logger.info("Artifacts uploaded.")


# ──────────────────────────────────────────────
# 3. Create the Databricks job
# ──────────────────────────────────────────────

def create_job() -> str:
    """Create the Structura parse_pdf job and return the job ID."""
    job_config = {
        "name": "structura-parse-pdf",
        "tasks": [
            {
                "task_key": "parse_pdf",
                "python_wheel_task": None,  # cleared below
                "spark_python_task": {
                    "python_file": f"{WORKSPACE_DIR}/parse_pdf.py",
                    "parameters": ["--s3_url", "{{job.parameters.s3_url}}"],
                },
                "new_cluster": {
                    "spark_version": "15.4.x-gpu-ml-scala2.12",
                    "node_type_id": "Standard_NC24ads_A100_v4",
                    "num_workers": 0,  # single-node
                    "spark_conf": {
                        "spark.master": "local[*]",
                        "spark.databricks.cluster.profile": "singleNode",
                    },
                    "custom_tags": {
                        "ResourceClass": "SingleNode",
                    },
                    "init_scripts": [
                        {"workspace": {"destination": f"{WORKSPACE_DIR}/init_script.sh"}},
                    ],
                },
                "libraries": [
                    {"pypi": {"package": "boto3"}},
                    {"pypi": {"package": "redis"}},
                    {"pypi": {"package": "pdf2image"}},
                    {"pypi": {"package": "google-genai"}},
                    {"pypi": {"package": "pillow"}},
                    {"pypi": {"package": "surya-ocr>=0.14.2"}},
                    {"pypi": {"package": "marker-pdf"}},
                ],
                "timeout_seconds": 1800,
            },
        ],
        "parameters": [
            {"name": "s3_url", "default": ""},
        ],
        "max_concurrent_runs": 10,
        "timeout_seconds": 1800,
    }

    # Remove the None placeholder
    del job_config["tasks"][0]["python_wheel_task"]

    result = api("POST", "/api/2.1/jobs/create", json=job_config)
    job_id = str(result["job_id"])
    logger.info(f"Created job: {job_id}")
    return job_id


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Structura Databricks Setup")
    print("=" * 60)
    print(f"Host: {HOST}")
    print()

    print("Step 1/3: Setting up secrets...")
    setup_secrets()
    print()

    print("Step 2/3: Uploading artifacts to Workspace...")
    upload_artifacts()
    print()

    print("Step 3/3: Creating Databricks job...")
    job_id = create_job()
    print()

    print("=" * 60)
    print("Setup complete!")
    print()
    print("Add the following to bpai/.env.prod:")
    print(f"  DATABRICKS_JOB_ID={job_id}")
    print()
    print("Test with:")
    print('  python -c "from bpai.databricks import submit_job; print(submit_job(\'test-url\'))"')
    print("=" * 60)


if __name__ == "__main__":
    main()
