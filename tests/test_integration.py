import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from app import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

def test_upload_endpoint_validation():
    with TestClient(app) as client:
        files = [("files", ("test.txt", b"dummy content", "text/plain"))]
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

def test_upload_endpoint_success():
    with TestClient(app) as client:
        files = [("files", ("test.pdf", b"a" * 100, "application/pdf"))]
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "Upload successful" in data["message"]

def test_ask_endpoint_success():
    with TestClient(app) as client:
        headers = {"X-Session-Id": "test-session-id"}
        payload = {"question": "What is Python?", "temperature": 0.2}
        response = client.post("/api/v1/ask", json=payload, headers=headers)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        # Read the event stream response
        body = response.text
        assert "data: [CONTEXT]:" in body
        assert "data: Answer " in body
        assert "data: content" in body
        assert "data: [DONE]" in body
