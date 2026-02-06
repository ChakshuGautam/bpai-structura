import logging
from core.db.client import supabase
from core.models.html_file import HTMLFile, HTMLFileCreate
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

class HTMLFileRepository:
    TABLE_NAME = "html_files"
    
    def _check_db_connection(self):
        """Check if the database connection is available."""
        logger.info(f"Checking database connection...")
        logger.info(f"Supabase client: {supabase}")
        logger.info(f"Supabase client type: {type(supabase)}")
        
        if not supabase:
            logger.warning("Supabase client not initialized. Database operations will fail.")
            raise ConnectionError("Database not configured")
        else:
            logger.info("Database connection check passed")
    
    async def create(self, html_file: HTMLFileCreate) -> HTMLFile:
        """Create a new HTML file record in the database."""
        try:
            logger.info(f"=== HTMLFileRepository.create() called ===")
            
            self._check_db_connection()
            
            # Log the input data
            data = html_file.model_dump()
            logger.info(f"Input HTMLFileCreate data:")
            logger.info(f"  user_id: {data.get('user_id')} (type: {type(data.get('user_id'))})")
            logger.info(f"  file_id: {data.get('file_id')} (type: {type(data.get('file_id'))})")
            logger.info(f"  content length: {len(data.get('content', ''))}")
            logger.info(f"  marker_output keys: {list(data.get('marker_output', {}).keys())}")
            logger.info(f"  metadata: {data.get('metadata')}")
            logger.info(f"  marker_version: {data.get('marker_version')}")
            logger.info(f"  status: {data.get('status')}")
            logger.info(f"  task_id: {data.get('task_id')}")
            logger.info(f"  modal_call_id: {data.get('modal_call_id')}")
            
            # Log the table name
            logger.info(f"Inserting into table: {self.TABLE_NAME}")
            
            # Perform the insert
            logger.info("Executing Supabase insert...")
            result = supabase.table(self.TABLE_NAME).insert(data).execute()
            
            # Log the result
            logger.info(f"Supabase insert result type: {type(result)}")
            logger.info(f"Result data: {result.data}")
            logger.info(f"Result count: {result.count}")
            
            # Check if we have any errors
            if hasattr(result, 'error') and result.error:
                logger.error(f"Supabase error: {result.error}")
                raise ValueError(f"Supabase error: {result.error}")
            
            if not result.data or len(result.data) == 0:
                logger.error("No data returned from Supabase insert")
                logger.error(f"Full result object: {result}")
                raise ValueError("Failed to create HTML file record - no data returned")
            
            # Log success
            logger.info(f"Successfully inserted record with ID: {result.data[0].get('id')}")
            
            # Validate and return the result
            html_file_result = HTMLFile.model_validate(result.data[0])
            logger.info(f"Successfully created HTMLFile object: {html_file_result.id}")
            
            return html_file_result
            
        except Exception as e:
            logger.error(f"=== ERROR IN HTMLFileRepository.create() ===")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.error(f"Full exception:", exc_info=True)
            raise ValueError(f"Database error: {str(e)}")

    async def get_by_file_id(self, file_id: str) -> HTMLFile:
        """Get HTML file by file_id for debugging."""
        try:
            logger.info(f"Searching for HTML file with file_id: {file_id}")
            result = supabase.table(self.TABLE_NAME).select("*").eq("file_id", file_id).execute()
            
            logger.info(f"Query result: {result.data}")
            
            if result.data and len(result.data) > 0:
                return HTMLFile.model_validate(result.data[0])
            else:
                logger.warning(f"No HTML file found with file_id: {file_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying HTML file: {str(e)}")
            raise

    async def list_recent(self, limit: int = 10) -> list[HTMLFile]:
        """List recent HTML files for debugging."""
        try:
            logger.info(f"Fetching {limit} most recent HTML files")
            result = supabase.table(self.TABLE_NAME).select("*").order("created_at", desc=True).limit(limit).execute()
            
            logger.info(f"Found {len(result.data)} HTML files")
            for file_data in result.data:
                logger.info(f"  - ID: {file_data.get('id')}, file_id: {file_data.get('file_id')}, created: {file_data.get('created_at')}")
            
            return [HTMLFile.model_validate(file_data) for file_data in result.data]
            
        except Exception as e:
            logger.error(f"Error listing HTML files: {str(e)}")
            raise

    async def get_by_user_and_file(self, user_id: UUID, file_id: UUID) -> Optional[HTMLFile]:
        """Get HTML file by user_id and file_id combination."""
        try:
            logger.info(f"Searching for HTML file with user_id: {user_id} and file_id: {file_id}")
            result = supabase.table(self.TABLE_NAME).select("*").eq("user_id", str(user_id)).eq("file_id", str(file_id)).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Found existing HTML file with ID: {result.data[0].get('id')}")
                return HTMLFile.model_validate(result.data[0])
            else:
                logger.info(f"No HTML file found for user_id: {user_id} and file_id: {file_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying HTML file by user and file IDs: {str(e)}")
            raise

    async def get_by_user_and_file_with_status(self, user_id: UUID, file_id: UUID) -> Optional[HTMLFile]:
        """Get HTML file by user_id and file_id, including status information."""
        try:
            logger.info(f"Searching for HTML file with user_id: {user_id} and file_id: {file_id} (with status)")
            result = supabase.table(self.TABLE_NAME).select("*").eq("user_id", str(user_id)).eq("file_id", str(file_id)).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Found existing HTML file with ID: {result.data[0].get('id')} and status: {result.data[0].get('status')}")
                return HTMLFile.model_validate(result.data[0])
            else:
                logger.info(f"No HTML file found for user_id: {user_id} and file_id: {file_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying HTML file by user and file IDs: {str(e)}")
            raise

    async def get_by_task_id(self, task_id: str) -> Optional[HTMLFile]:
        """Get HTML file by task_id."""
        try:
            logger.info(f"Searching for HTML file with task_id: {task_id}")
            result = supabase.table(self.TABLE_NAME).select("*").eq("task_id", task_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Found existing HTML file with ID: {result.data[0].get('id')} and status: {result.data[0].get('status')}")
                return HTMLFile.model_validate(result.data[0])
            else:
                logger.info(f"No HTML file found with task_id: {task_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying HTML file by task_id: {str(e)}")
            raise

    async def update_status(self, html_file_id: UUID, status: str, task_id: Optional[UUID] = None, modal_call_id: Optional[str] = None):
        """Update the status of an HTML file."""
        try:
            logger.info(f"Updating HTML file {html_file_id} status to: {status}")
            update_data = {"status": status}
            if task_id:
                update_data["task_id"] = str(task_id)
            if modal_call_id:
                update_data["modal_call_id"] = modal_call_id
                
            result = supabase.table(self.TABLE_NAME).update(update_data).eq("id", str(html_file_id)).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully updated HTML file status")
                return result.data[0]
            else:
                logger.error("No data returned from status update")
                return None
                
        except Exception as e:
            logger.error(f"Error updating HTML file status: {str(e)}")
            raise