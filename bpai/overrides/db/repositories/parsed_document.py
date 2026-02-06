import json
import logging
from uuid import UUID
from typing import List, Optional

from psycopg2.extras import RealDictCursor

from core.db.client import get_connection, put_connection
from core.models.parsed_document import ParsedDocument, ParsedDocumentCreate, ParsedDocumentUpdate

logger = logging.getLogger(__name__)


class ParsedDocumentRepository:
    TABLE_NAME = "parsed_documents"

    def _conn(self):
        return get_connection()

    def _put(self, conn):
        put_connection(conn)

    async def create(self, parsed_document: ParsedDocumentCreate) -> ParsedDocument:
        conn = self._conn()
        try:
            data = parsed_document.model_dump()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO parsed_documents
                        (user_id, document_type_id, data, original_file_name, original_file_url, file_id)
                    VALUES
                        (%(user_id)s, %(document_type_id)s, %(data)s,
                         %(original_file_name)s, %(original_file_url)s, %(file_id)s)
                    RETURNING *
                    """,
                    {
                        "user_id": str(data["user_id"]),
                        "document_type_id": str(data["document_type_id"]),
                        "data": json.dumps(data["data"]),
                        "original_file_name": data["original_file_name"],
                        "original_file_url": data.get("original_file_url"),
                        "file_id": str(data["file_id"]) if data.get("file_id") else None,
                    },
                )
                row = cur.fetchone()
            conn.commit()
            return ParsedDocument.model_validate(dict(row))
        except Exception as e:
            conn.rollback()
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)

    async def get_by_id(self, parsed_document_id: UUID) -> Optional[ParsedDocument]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM parsed_documents WHERE id = %s",
                    (str(parsed_document_id),),
                )
                row = cur.fetchone()
            return ParsedDocument.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def list_by_user(self, user_id: UUID) -> List[ParsedDocument]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM parsed_documents WHERE user_id = %s",
                    (str(user_id),),
                )
                rows = cur.fetchall()
            return [ParsedDocument.model_validate(dict(r)) for r in rows]
        finally:
            self._put(conn)

    async def list_by_document_type(self, document_type_id: UUID) -> List[ParsedDocument]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM parsed_documents WHERE document_type_id = %s",
                    (str(document_type_id),),
                )
                rows = cur.fetchall()
            return [ParsedDocument.model_validate(dict(r)) for r in rows]
        finally:
            self._put(conn)

    async def update(self, parsed_document_id: UUID, parsed_document: ParsedDocumentUpdate) -> Optional[ParsedDocument]:
        conn = self._conn()
        try:
            update_data = {k: v for k, v in parsed_document.model_dump().items() if v is not None}
            if not update_data:
                return await self.get_by_id(parsed_document_id)

            sets = []
            vals = []
            for k, v in update_data.items():
                sets.append(f"{k} = %s")
                vals.append(json.dumps(v) if k == "data" else v)
            sets.append("updated_at = NOW()")
            vals.append(str(parsed_document_id))

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"UPDATE parsed_documents SET {', '.join(sets)} WHERE id = %s RETURNING *",
                    vals,
                )
                row = cur.fetchone()
            conn.commit()
            return ParsedDocument.model_validate(dict(row)) if row else None
        except Exception as e:
            conn.rollback()
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)

    async def delete(self, parsed_document_id: UUID) -> bool:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM parsed_documents WHERE id = %s RETURNING id",
                    (str(parsed_document_id),),
                )
                deleted = cur.fetchone()
            conn.commit()
            return deleted is not None
        except Exception as e:
            conn.rollback()
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)

    async def get_by_document_type_and_filename(
        self, document_type_id: UUID, user_id: UUID, original_file_name: str
    ) -> Optional[ParsedDocument]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM parsed_documents
                    WHERE document_type_id = %s AND user_id = %s AND original_file_name = %s
                    """,
                    (str(document_type_id), str(user_id), original_file_name),
                )
                row = cur.fetchone()
            return ParsedDocument.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def delete_by_document_type(self, document_type_id: UUID) -> bool:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM parsed_documents WHERE document_type_id = %s RETURNING id",
                    (str(document_type_id),),
                )
                deleted = cur.fetchall()
            conn.commit()
            return len(deleted) > 0
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error deleting parsed documents by document type: {e}")
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)
