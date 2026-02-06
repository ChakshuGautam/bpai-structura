"""
Client-side Databricks job submission module.

Used by the FastAPI backend to submit PDF processing jobs to Databricks
instead of Modal. Provides submit_job() and get_job_status().
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_JOB_ID = os.environ.get("DATABRICKS_JOB_ID")


def _get_headers() -> dict:
    token = DATABRICKS_TOKEN or os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise ValueError("DATABRICKS_TOKEN environment variable is not set")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _get_host() -> str:
    host = DATABRICKS_HOST or os.environ.get("DATABRICKS_HOST")
    if not host:
        raise ValueError("DATABRICKS_HOST environment variable is not set")
    return host.rstrip("/")


def _get_job_id() -> str:
    job_id = DATABRICKS_JOB_ID or os.environ.get("DATABRICKS_JOB_ID")
    if not job_id:
        raise ValueError("DATABRICKS_JOB_ID environment variable is not set")
    return job_id


def submit_job(s3_url_with_task_id: str) -> str:
    """
    Submit a PDF processing job to Databricks.

    Calls the Databricks Jobs API POST /api/2.1/jobs/run-now to trigger
    the parse_pdf job with the given S3 URL.

    Args:
        s3_url_with_task_id: S3 URL of the uploaded PDF with task_id appended.

    Returns:
        run_id: The Databricks run ID for tracking the job.
    """
    host = _get_host()
    url = f"{host}/api/2.1/jobs/run-now"

    payload = {
        "job_id": int(_get_job_id()),
        "python_named_params": {
            "s3_url": s3_url_with_task_id,
        },
    }

    logger.info(f"Submitting Databricks job: job_id={payload['job_id']}, s3_url={s3_url_with_task_id}")

    response = requests.post(url, headers=_get_headers(), json=payload)
    response.raise_for_status()

    data = response.json()
    run_id = str(data["run_id"])
    logger.info(f"Databricks job submitted: run_id={run_id}")
    return run_id


def get_job_status(run_id: str) -> dict:
    """
    Get the status of a Databricks job run.

    Calls the Databricks Jobs API GET /api/2.1/jobs/runs/get.

    Args:
        run_id: The Databricks run ID.

    Returns:
        dict with keys:
            - state: The run lifecycle state (PENDING, RUNNING, TERMINATED, etc.)
            - result_state: The result state if terminated (SUCCESS, FAILED, etc.)
            - status_message: Human-readable status message
    """
    host = _get_host()
    url = f"{host}/api/2.1/jobs/runs/get"

    params = {"run_id": run_id}

    logger.info(f"Checking Databricks job status: run_id={run_id}")

    response = requests.get(url, headers=_get_headers(), params=params)
    response.raise_for_status()

    data = response.json()
    state = data.get("state", {})

    return {
        "state": state.get("life_cycle_state", "UNKNOWN"),
        "result_state": state.get("result_state"),
        "status_message": state.get("state_message", ""),
    }
