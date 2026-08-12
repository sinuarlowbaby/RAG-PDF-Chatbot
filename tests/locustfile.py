from locust import HttpUser, task, between
import os

class RAGChatbotUser(HttpUser):
    wait_time = between(2.0, 5.0)
    
    # ⚠️ Replace this with a real session_id you got after uploading a document on your server!
    # Otherwise, the server will return a 404 (Session not found) or 422 (Unprocessable Entity)
    SESSION_ID = os.environ.get("LOCUST_SESSION_ID", "4f2d7828-08d8-421e-a575-cd550d0ff5a7")

    @task
    def ask_question(self):
        payload = {
            "question": "What is the main topic of the document?",
            "temperature": 0.5,
            "max_tokens": 500,
            "use_answer_cache": True,
            "use_multi_query_cache": True
        }
        
        headers = {
            "x-session-id": self.SESSION_ID
        }
        
        # Testing the streaming endpoint
        with self.client.post("/api/v1/ask", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 202:
                # 202 means the document is still processing in the background, not technically a failure of the API logic
                response.failure("202: Document is still processing")
            elif response.status_code == 404:
                response.failure("404: Session not found! Did you use a real session_id?")
            elif response.status_code == 429:
                response.failure("429: Rate limited! (Too many requests)")
            else:
                response.failure(f"Failed with status {response.status_code}")
