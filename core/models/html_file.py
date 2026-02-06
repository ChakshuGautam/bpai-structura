from datetime import datetime
from uuid import UUID
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json

class JSONSerializableMixin:
    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        return self._serialize_uuids(d)

    def _serialize_uuids(self, data):
        if isinstance(data, dict):
            return {k: self._serialize_uuids(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_uuids(item) for item in data]
        elif isinstance(data, UUID):
            return str(data)
        return data

class HTMLFileCreate(BaseModel, JSONSerializableMixin):
    user_id: UUID
    file_id: UUID
    marker_output: Dict[str, Any]
    marker_version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None
    status: str = "pending"
    task_id: Optional[UUID] = None
    # modal_call_id: Optional[str] = None

    class Config:
        json_encoders = {
            UUID: str
        }

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        return self._serialize_uuids(data)

class HTMLFile(HTMLFileCreate):
    id: UUID
    created_at: datetime  # Move to this class
    updated_at: datetime