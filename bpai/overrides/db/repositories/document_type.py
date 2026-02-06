import json
import logging
from uuid import UUID
from typing import List, Optional

from psycopg2.extras import RealDictCursor

from core.db.client import get_connection, put_connection
from core.models.document_type import DocumentType, DocumentTypeCreate, DocumentTypeUpdate

logger = logging.getLogger(__name__)


class DocumentTypeRepository:
    TABLE_NAME = "document_types"

    def _conn(self):
        return get_connection()

    def _put(self, conn):
        put_connection(conn)

    async def create(self, document_type: DocumentTypeCreate) -> DocumentType:
        conn = self._conn()
        try:
            data = document_type.model_dump()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO document_types (user_id, type, description, schema)
                    VALUES (%(user_id)s, %(type)s, %(description)s, %(schema)s)
                    RETURNING *
                    """,
                    {
                        "user_id": str(data["user_id"]),
                        "type": data["type"],
                        "description": data["description"],
                        "schema": json.dumps(data["schema"]),
                    },
                )
                row = cur.fetchone()
            conn.commit()
            return DocumentType.model_validate(dict(row))
        except Exception as e:
            conn.rollback()
            err = str(e)
            if "document_types_user_id_type_key" in err:
                raise ValueError("A document type with the same name already exists for this user")
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)

    async def get_by_id(self, document_type_id: UUID) -> Optional[DocumentType]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM document_types WHERE id = %s",
                    (str(document_type_id),),
                )
                row = cur.fetchone()
            return DocumentType.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def get_by_type_and_user(self, type_name: str, user_id: UUID) -> Optional[DocumentType]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM document_types WHERE type = %s AND user_id = %s",
                    (type_name, str(user_id)),
                )
                row = cur.fetchone()
            return DocumentType.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def list_by_user(self, user_id: UUID) -> List[DocumentType]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM document_types WHERE user_id = %s",
                    (str(user_id),),
                )
                rows = cur.fetchall()
            return [DocumentType.model_validate(dict(r)) for r in rows]
        finally:
            self._put(conn)

    async def list_all(self) -> List[DocumentType]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM document_types")
                rows = cur.fetchall()
            return [DocumentType.model_validate(dict(r)) for r in rows]
        finally:
            self._put(conn)

    async def update(self, document_type_id: UUID, document_type: DocumentTypeUpdate) -> Optional[DocumentType]:
        conn = self._conn()
        try:
            update_data = {k: v for k, v in document_type.model_dump().items() if v is not None}
            if not update_data:
                return await self.get_by_id(document_type_id)

            sets = []
            vals = []
            for k, v in update_data.items():
                sets.append(f"{k} = %s")
                vals.append(json.dumps(v) if k == "schema" else v)
            vals.append(str(document_type_id))

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"UPDATE document_types SET {', '.join(sets)} WHERE id = %s RETURNING *",
                    vals,
                )
                row = cur.fetchone()
            conn.commit()
            return DocumentType.model_validate(dict(row)) if row else None
        except Exception as e:
            conn.rollback()
            err = str(e)
            if "document_types_user_id_type_key" in err:
                raise ValueError("Cannot update: another document type with the same name exists for this user")
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)

    async def delete(self, document_type_id: UUID) -> bool:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_types WHERE id = %s RETURNING id",
                    (str(document_type_id),),
                )
                deleted = cur.fetchone()
            conn.commit()
            return deleted is not None
        except Exception as e:
            conn.rollback()
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)
