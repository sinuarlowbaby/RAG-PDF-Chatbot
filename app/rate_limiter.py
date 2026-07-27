from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

def custom_key_func(request: Request) -> str:
    """
    Tries to rate limit by Session ID first. 
    If no Session ID is present, falls back to IP address.
    """
    session_id = request.headers.get("x-session-id")
    if session_id:
        return f"session:{session_id}"
    return get_remote_address(request)

limiter = Limiter(key_func=custom_key_func, default_limits=["20/minute"])
