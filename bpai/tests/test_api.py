"""
BPAI API integration tests.
Run from inside the API container or from the VM with access to localhost:8000.

Usage:
    python test_api.py
"""
import os
import sys
import json
import uuid
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_KEY = os.environ.get("STRUCTURA_API_KEY", "Ngxz4MjjSEiKoqfzWIsmLSlHyvoDPpZ-5nmjKUvJ1M8")
HEADERS = {"X-Api-Key": API_KEY}
JSON_HEADERS = {**HEADERS, "Content-Type": "application/json"}

TEST_USER_ID = str(uuid.uuid4())
passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1
        errors.append((name, str(e)))
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        failed += 1
        errors.append((name, f"{type(e).__name__}: {e}"))


# ── Auth tests ──────────────────────────────────────────────────────────
print("\n=== Auth Tests ===")


def test_missing_api_key():
    r = requests.get(f"{API_BASE}/api/v1/document-types", params={"user_id": TEST_USER_ID})
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"


def test_wrong_api_key():
    r = requests.get(
        f"{API_BASE}/api/v1/document-types",
        params={"user_id": TEST_USER_ID},
        headers={"X-Api-Key": "wrong-key"},
    )
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"


test("Missing API key returns 401", test_missing_api_key)
test("Wrong API key returns 401", test_wrong_api_key)

# ── Document Type CRUD ──────────────────────────────────────────────────
print("\n=== Document Type CRUD ===")
doc_type_id = None


def test_create_document_type():
    global doc_type_id
    r = requests.post(
        f"{API_BASE}/api/v1/document-types",
        headers=JSON_HEADERS,
        json={
            "user_id": TEST_USER_ID,
            "type": "TestInvoice",
            "description": "Test invoice type",
            "schema": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "amount": {"type": "number"},
                },
            },
        },
    )
    assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["success"] is True
    assert body["data"]["type"] == "TestInvoice"
    assert body["data"]["user_id"] == TEST_USER_ID
    doc_type_id = body["data"]["id"]


def test_list_document_types():
    r = requests.get(
        f"{API_BASE}/api/v1/document-types",
        params={"user_id": TEST_USER_ID},
        headers=HEADERS,
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["success"] is True
    assert len(body["data"]) >= 1
    types = [d["type"] for d in body["data"]]
    assert "TestInvoice" in types


def test_get_document_type_by_id():
    r = requests.get(f"{API_BASE}/api/v1/document-types/{doc_type_id}", headers=HEADERS)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["data"]["id"] == doc_type_id
    assert body["data"]["type"] == "TestInvoice"


def test_get_document_type_by_user_and_type():
    r = requests.get(
        f"{API_BASE}/api/v1/document-types/{TEST_USER_ID}/TestInvoice",
        headers=HEADERS,
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["data"]["document_type_id"] == doc_type_id


def test_duplicate_create_returns_409():
    r = requests.post(
        f"{API_BASE}/api/v1/document-types",
        headers=JSON_HEADERS,
        json={
            "user_id": TEST_USER_ID,
            "type": "TestInvoice",
            "description": "Duplicate",
            "schema": {},
        },
    )
    assert r.status_code == 409, f"Expected 409, got {r.status_code}: {r.text}"


def test_update_document_type():
    r = requests.put(
        f"{API_BASE}/api/v1/document-types/{doc_type_id}",
        headers=JSON_HEADERS,
        json={"description": "Updated description"},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["data"]["description"] == "Updated description"


def test_patch_schema():
    new_schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "amount": {"type": "number"},
            "date": {"type": "string"},
        },
    }
    r = requests.patch(
        f"{API_BASE}/api/v1/document-types/{doc_type_id}/schema",
        headers=JSON_HEADERS,
        json={"schema": new_schema},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert "date" in body["data"]["schema"]["properties"]


test("Create document type", test_create_document_type)
test("List document types", test_list_document_types)
test("Get document type by ID", test_get_document_type_by_id)
test("Get document type by user+type", test_get_document_type_by_user_and_type)
test("Duplicate create returns 409", test_duplicate_create_returns_409)
test("Update document type", test_update_document_type)
test("Patch schema", test_patch_schema)

# ── Convert v2 validation ───────────────────────────────────────────────
print("\n=== Convert v2 Validation ===")


def test_v2_missing_input():
    r = requests.post(
        f"{API_BASE}/api/v2/convert",
        headers=HEADERS,
        data={"user_id": TEST_USER_ID, "file_id": str(uuid.uuid4())},
    )
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"


def test_v2_wrong_content_type():
    r = requests.post(
        f"{API_BASE}/api/v2/convert",
        headers=HEADERS,
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
        data={"user_id": TEST_USER_ID, "file_id": str(uuid.uuid4())},
    )
    assert r.status_code == 415, f"Expected 415, got {r.status_code}: {r.text}"


def test_v2_invalid_modal_app():
    r = requests.post(
        f"{API_BASE}/api/v2/convert",
        headers=HEADERS,
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"modal_app": "invalid"},
    )
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"


def test_v2_get_nonexistent_task():
    fake_task = str(uuid.uuid4())
    r = requests.get(f"{API_BASE}/api/v2/convert/{fake_task}", headers=HEADERS)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["status"] == "processing"


def test_v2_get_html_nonexistent():
    r = requests.get(
        f"{API_BASE}/api/v2/convert/html/{TEST_USER_ID}/{uuid.uuid4()}",
        headers=HEADERS,
    )
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"


test("v2 missing file and url returns 422", test_v2_missing_input)
test("v2 wrong content type returns 415", test_v2_wrong_content_type)
test("v2 invalid modal_app returns 422", test_v2_invalid_modal_app)
test("v2 nonexistent task returns processing", test_v2_get_nonexistent_task)
test("v2 nonexistent html returns 404", test_v2_get_html_nonexistent)

# ── Convert v3 validation ───────────────────────────────────────────────
print("\n=== Convert v3 Validation ===")


def test_v3_missing_input():
    r = requests.post(f"{API_BASE}/api/v3/convert", headers=HEADERS)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"


test("v3 missing input returns 422", test_v3_missing_input)

# ── Cleanup ─────────────────────────────────────────────────────────────
print("\n=== Cleanup ===")


def test_delete_document_type():
    r = requests.delete(f"{API_BASE}/api/v1/document-types/{doc_type_id}", headers=HEADERS)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["data"] is True


def test_get_deleted_returns_404():
    r = requests.get(f"{API_BASE}/api/v1/document-types/{doc_type_id}", headers=HEADERS)
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"


test("Delete document type", test_delete_document_type)
test("Get deleted type returns 404", test_get_deleted_returns_404)

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if errors:
    print("\nFailures:")
    for name, err in errors:
        print(f"  - {name}: {err}")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
