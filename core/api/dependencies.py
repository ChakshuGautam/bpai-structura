from fastapi import Header, HTTPException, status
import os

# API key verification
API_KEY_ENV = "STRUCTURA_API_KEY"


async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Verify API key middleware."""
    expected_key = os.environ.get(API_KEY_ENV)
    
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server"
        )
    
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    return True