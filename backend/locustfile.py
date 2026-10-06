from locust import HttpUser, task, between
import random

class AstrologyChatbotUser(HttpUser):
    # Set host explicitly so Locust web UI defaults to this exact URL
    host = "http://127.0.0.1:8000"
    wait_time = between(5, 12)  # Pause 5 to 12 seconds between questions

    @task
    def send_chat_request(self):
        user_id = f"user_{random.randint(1, 10000)}"
        payload = {
            "user_id": user_id,
            "name": "Krishna",
            "query": "What are my career prospects based on my planetary positions?",
            "birth_year": 2006,
            "birth_month": 7,
            "birth_day": 22,
            "birth_hour": 13.05,
            "latitude": 21.3847,
            "longitude": 78.9188
        }
        headers = {"Content-Type": "application/json"}
        
        self.client.post("/chat", json=payload, headers=headers)