import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from app import app

from rate_limiter import limiter

@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter state before each test so previous tests don't pollute limits."""
    limiter.reset()

def test_ip_limit_on_upload():
    """Test that the /upload endpoint rate limits by IP (2 per minute)."""
    with TestClient(app) as client:
        files = [("files", ("test.pdf", b"dummy content", "application/pdf"))]
        for _ in range(2):
            response = client.post("/api/v1/upload", files=files)
            assert response.status_code == 200

        # The 3rd request should trigger the rate limit
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 429
        assert "2 per 1 minute" in response.text

def test_session_limit_on_ask():
    """Test that the /ask endpoint rate limits by session ID (5 per minute)."""
    with TestClient(app) as client:
        session_id = "test-session-123"
        headers = {"x-session-id": session_id}
        payload = {"question": "Hello", "temperature": 0.2}
        
        # We will send 5 requests. 
        for _ in range(5):
            response = client.post("/api/v1/ask", headers=headers, json=payload)
            assert response.status_code == 200

        # The 6th request with the SAME session_id should trigger the rate limit
        response = client.post("/api/v1/ask", headers=headers, json=payload)
        assert response.status_code == 429
        assert "5 per 1 minute" in response.text

def test_different_sessions_on_ask():
    """Test that different session IDs do not trigger the session limit."""
    with TestClient(app) as client:
        for i in range(6):
            session_id = f"unique-session-{i}"
            headers = {"x-session-id": session_id}
            payload = {"question": "Hello", "temperature": 0.2}
            
            response = client.post("/api/v1/ask", headers=headers, json=payload)
            assert response.status_code == 200

def test_attack_session_bruteforce_with_ip_rotation():
    """
    ATTACK SIMULATION: 
    An attacker knows a target's session ID and tries to spam it (or drain their LLM tokens), 
    while constantly rotating their IP address to bypass traditional IP bans.
    
    EXPECTED: Because our custom limiter prioritizes Session ID, 
    the attack should still be blocked after 5 requests, regardless of the spoofed IP!
    """
    with TestClient(app) as client:
        target_session = "victim-session-999"
        
        # Send 5 requests, each from a "different" IP address
        for i in range(5):
            headers = {
                "x-session-id": target_session,
                "X-Forwarded-For": f"10.0.0.{i}" # Attacker rotating IP
            }
            payload = {"question": "spam", "temperature": 0.2}
            response = client.post("/api/v1/ask", headers=headers, json=payload)
            assert response.status_code == 200
            
        # The 6th request with a brand NEW IP but the SAME target session ID should be blocked!
        headers = {
            "x-session-id": target_session,
            "X-Forwarded-For": "199.199.199.199" # Completely new IP
        }
        payload = {"question": "spam", "temperature": 0.2}
        response = client.post("/api/v1/ask", headers=headers, json=payload)
        
        assert response.status_code == 429
        assert "5 per 1 minute" in response.text


