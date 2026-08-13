import os
import random
import time
from pathlib import Path
from locust import HttpUser, task, between, events

# ── Built-in in-memory valid PDF bytes for autonomous testing ──────────────────
DEFAULT_SAMPLE_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Length 400 >>
stream
BT
/F1 12 Tf
50 720 Td
(Retrieval Augmented Generation Benchmark Document) Tj
0 -25 Td
(This document is automatically generated for comprehensive load and performance testing.) Tj
0 -20 Td
(It covers hybrid retrieval combining dense vector embeddings and BM25 full text search.) Tj
0 -20 Td
(FastAPI handles asynchronous streaming responses with Gunicorn and Uvicorn workers.) Tj
0 -20 Td
(Redis semantic cache significantly reduces response latency for repetitive queries.) Tj
0 -20 Td
(Qdrant vector database provides scalable similarity search for chunked text embeddings.) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000280 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
600
%%EOF"""

QUESTION_POOL = [
    "What is the main purpose of this benchmark document?",
    "How does hybrid retrieval work according to the text?",
    "What role does Redis play in this system?",
    "How does Qdrant fit into the architecture?",
    "Explain the server setup using FastAPI and Gunicorn.",
]

CACHE_TEST_QUESTION = "What is the main purpose of this benchmark document?"


class RAGFullJourneyUser(HttpUser):
    """
    Simulates a complete real-world user journey:
    1. Uploads a real PDF document on startup.
    2. Waits for the background vector ingestion to complete.
    3. Asks multiple questions (testing cold LLM queries and warm semantic cache hits).
    4. Cleans up its session and vector points on disconnect.
    """
    # Simulate realistic human reading / typing time (2 to 6 seconds between actions)
    wait_time = between(2.0, 6.0)

    def on_start(self):
        """Executed when a simulated user spawns."""
        self.session_id = None
        
        # Check if user specified a fixed existing session via environment variable
        fixed_session = os.environ.get("LOCUST_SESSION_ID")
        if fixed_session:
            self.session_id = fixed_session
            return

        # Otherwise, upload a real PDF document dynamically
        pdf_path = os.environ.get("LOCUST_PDF_PATH")
        if pdf_path and Path(pdf_path).exists():
            pdf_bytes = Path(pdf_path).read_bytes()
            filename = Path(pdf_path).name
        else:
            pdf_bytes = DEFAULT_SAMPLE_PDF
            filename = f"benchmark_test_{random.randint(1000, 9999)}.pdf"

        files = {
            "files": (filename, pdf_bytes, "application/pdf")
        }

        with self.client.post("/api/v1/upload", files=files, name="1. [Upload] Ingest PDF", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data.get("session_id")
                resp.success()
            else:
                resp.failure(f"Upload failed with status {resp.status_code}: {resp.text}")
                return

        # Poll until the document is ingested and ready for querying
        self._wait_for_ingestion()

    def _wait_for_ingestion(self, max_retries=10, retry_delay=2.0):
        """Wait until background ingestion completes (202 -> 200)."""
        if not self.session_id:
            return

        headers = {"x-session-id": self.session_id}
        payload = {
            "question": "warmup test",
            "temperature": 0.2,
            "use_answer_cache": False,
            "use_multi_query_cache": False,
        }

        for _ in range(max_retries):
            time.sleep(retry_delay)
            with self.client.post(
                "/api/v1/ask",
                json=payload,
                headers=headers,
                name="1.1 [Ingest] Ingestion Readiness Probe",
                catch_response=True,
            ) as probe_resp:
                if probe_resp.status_code == 200:
                    probe_resp.success()
                    return
                elif probe_resp.status_code == 202:
                    # Still processing in background, wait and retry
                    probe_resp.success()
                else:
                    probe_resp.failure(f"Readiness check returned {probe_resp.status_code}")
                    return

    @task(5)
    def ask_unique_question(self):
        """Simulates querying new, uncached questions (tests Qdrant search + LLM generation)."""
        if not self.session_id:
            return

        question = random.choice(QUESTION_POOL)
        payload = {
            "question": question,
            "temperature": 0.5,
            "max_tokens": 500,
            "use_answer_cache": True,
            "use_multi_query_cache": True,
        }
        headers = {"x-session-id": self.session_id}

        with self.client.post(
            "/api/v1/ask",
            json=payload,
            headers=headers,
            name="2. [Chat] Ask Question (Uncached / Search)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.failure("429: Rate limited by server")
            else:
                resp.failure(f"Chat failed with status {resp.status_code}")

    @task(3)
    def ask_cached_question(self):
        """Simulates asking repeated questions to stress-test Redis Semantic Cache hit rate & speed."""
        if not self.session_id:
            return

        payload = {
            "question": CACHE_TEST_QUESTION,
            "temperature": 0.5,
            "max_tokens": 500,
            "use_answer_cache": True,
            "use_multi_query_cache": True,
        }
        headers = {"x-session-id": self.session_id}

        with self.client.post(
            "/api/v1/ask",
            json=payload,
            headers=headers,
            name="3. [Chat] Ask Repeated (Semantic Cache Hit)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.failure("429: Rate limited by server")
            else:
                resp.failure(f"Cache test failed with status {resp.status_code}")

    @task(1)
    def check_health(self):
        """Simulates load-balancer health checks."""
        self.client.get("/health", name="0. [System] Health Check")

    def on_stop(self):
        """Clean up vector points and Redis cache when a user finishes testing."""
        if self.session_id and not os.environ.get("LOCUST_SESSION_ID"):
            self.client.delete(
                f"/api/v1/session/{self.session_id}",
                name="4. [Cleanup] Delete Session Vectors",
            )
