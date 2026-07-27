import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from app import app

# Initialize TestClient
client = TestClient(app)

def test_ip_limit_on_upload():
    """Test that the /upload endpoint rate limits by IP (2 per minute)."""
    # The limit is 2/minute.
    # We will send 3 requests. The 3rd should fail with 429.
    
    for i in range(2):
        # We don't need valid files to trigger rate limiter
        response = client.post("/upload")
        # Should be 400 or 422 because we didn't provide files, but NOT 429
        assert response.status_code != 429

    # The 3rd request should trigger the rate limit
    response = client.post("/upload")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text

def test_session_limit_on_ask():
    """Test that the /ask endpoint rate limits by session ID (5 per minute)."""
    session_id = "test-session-123"
    headers = {"x-session-id": session_id}
    payload = {"question": "Hello", "chat_history": []}
    
    # We will send 5 requests. 
    # They should not be 429 (might be 400/404/500 depending on mock data, but not 429)
    for i in range(5):
        response = client.post("/ask", headers=headers, json=payload)
        assert response.status_code != 429

    # The 6th request with the SAME session_id should trigger the rate limit
    response = client.post("/ask", headers=headers, json=payload)
    assert response.status_code == 429

def test_different_sessions_on_ask():
    """Test that different session IDs do not trigger the session limit."""
    # We will send 6 requests, but each with a DIFFERENT session ID.
    # Because our limiter groups by session_id, they should ALL pass the 5/min session limit.
    # (Note: They will eventually hit the global 20/min IP limit if we run too many, but 6 is fine).
    
    for i in range(6):
        session_id = f"unique-session-{i}"
        headers = {"x-session-id": session_id}
        payload = {"question": "Hello", "chat_history": []}
        
        response = client.post("/ask", headers=headers, json=payload)
        # Should not trigger a 429
        assert response.status_code != 429

def test_attack_session_bruteforce_with_ip_rotation():
    """
    ATTACK SIMULATION: 
    An attacker knows a target's session ID and tries to spam it (or drain their LLM tokens), 
    while constantly rotating their IP address to bypass traditional IP bans.
    
    EXPECTED: Because our custom limiter prioritizes Session ID, 
    the attack should still be blocked after 5 requests, regardless of the spoofed IP!
    """
    target_session = "victim-session-999"
    
    # Send 5 requests, each from a "different" IP address
    for i in range(5):
        headers = {
            "x-session-id": target_session,
            "X-Forwarded-For": f"10.0.0.{i}" # Attacker rotating IP
        }
        payload = {"question": "spam", "chat_history": []}
        response = client.post("/ask", headers=headers, json=payload)
        assert response.status_code != 429
        
    # The 6th request with a brand NEW IP but the SAME target session ID should be blocked!
    headers = {
        "x-session-id": target_session,
        "X-Forwarded-For": "199.199.199.199" # Completely new IP
    }
    payload = {"question": "spam", "chat_history": []}
    response = client.post("/ask", headers=headers, json=payload)
    
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text
