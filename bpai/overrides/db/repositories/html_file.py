import json
import logging
from typing import Optional
from uuid import UUID

from psycopg2.extras import RealDictCursor

from core.db.client import get_connection, put_connection
from core.models.html_file import HTMLFile, HTMLFileCreate

logger = logging.getLogger(__name__)


class HTMLFileRepository:
    TABLE_NAME = "html_files"

    def _conn(self):
        return get_connection()

    def _put(self, conn):
        put_connection(conn)

    async def create(self, html_file: HTMLFileCreate) -> HTMLFile:
        conn = self._conn()
        try:
            data = html_file.model_dump()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO html_files
                        (user_id, file_id, marker_output, marker_version, metadata, status, task_id)
                    VALUES
                        (%(user_id)s, %(file_id)s, %(marker_output)s, %(marker_version)s,
                         %(metadata)s, %(status)s, %(task_id)s)
                    RETURNING *
                    """,
                    {
                        "user_id": str(data["user_id"]),
                        "file_id": str(data["file_id"]),
                        "marker_output": json.dumps(data.get("marker_output", {})),
                        "marker_version": data.get("marker_version", "1.0"),
                        "metadata": json.dumps(data["metadata"]) if data.get("metadata") else None,
                        "status": data.get("status", "pending"),
                        "task_id": str(data["task_id"]) if data.get("task_id") else None,
                    },
                )
                row = cur.fetchone()
            conn.commit()
            logger.info(f"Created html_file record: {row['id']}")
            return HTMLFile.model_validate(dict(row))
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in HTMLFileRepository.create(): {e}", exc_info=True)
            raise ValueError(f"Database error: {e}")
        finally:
            self._put(conn)

    async def get_by_file_id(self, file_id: str) -> Optional[HTMLFile]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM html_files WHERE file_id = %s", (str(file_id),)
                )
                row = cur.fetchone()
            return HTMLFile.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def list_recent(self, limit: int = 10) -> list[HTMLFile]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM html_files ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
            return [HTMLFile.model_validate(dict(r)) for r in rows]
        finally:
            self._put(conn)

    async def get_by_user_and_file(self, user_id: UUID, file_id: UUID) -> Optional[HTMLFile]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM html_files WHERE user_id = %s AND file_id = %s",
                    (str(user_id), str(file_id)),
                )
                row = cur.fetchone()
            return HTMLFile.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def get_by_user_and_file_with_status(self, user_id: UUID, file_id: UUID) -> Optional[HTMLFile]:
        return await self.get_by_user_and_file(user_id, file_id)

    async def get_by_task_id(self, task_id: str) -> Optional[HTMLFile]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM html_files WHERE task_id = %s", (str(task_id),)
                )
                row = cur.fetchone()
            return HTMLFile.model_validate(dict(row)) if row else None
        finally:
            self._put(conn)

    async def update_status(
        self,
        html_file_id: UUID,
        status: str,
        task_id: Optional[UUID] = None,
        modal_call_id: Optional[str] = None,
    ):
        conn = self._conn()
        try:
            sets = ["status = %s"]
            vals: list = [status]
            if task_id:
                sets.append("task_id = %s")
                vals.append(str(task_id))
            vals.append(str(html_file_id))

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"UPDATE html_files SET {', '.join(sets)} WHERE id = %s RETURNING *",
                    vals,
                )
                row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating html_file status: {e}")
            raise
        finally:
            self._put(conn)

    async def update_marker_output(self, html_file_id: UUID, marker_output: dict, metadata: dict = None):
        """Update marker_output and optionally metadata for an html_file record."""
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if metadata is not None:
                    cur.execute(
                        "UPDATE html_files SET marker_output = %s, metadata = %s WHERE id = %s RETURNING *",
                        (json.dumps(marker_output), json.dumps(metadata), str(html_file_id)),
                    )
                else:
                    cur.execute(
                        "UPDATE html_files SET marker_output = %s WHERE id = %s RETURNING *",
                        (json.dumps(marker_output), str(html_file_id)),
                    )
                row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating marker_output: {e}")
            raise
        finally:
            self._put(conn)
