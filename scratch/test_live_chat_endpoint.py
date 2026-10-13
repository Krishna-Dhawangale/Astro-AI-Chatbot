import requests
import json

url = "http://127.0.0.1:8000/chat"

payload = {
    "query": "kab milegi job",
    "user_id": "test_user_micro_1",
    "birth_details": {
        "year": 1995,
        "month": 5,
        "day": 15,
        "hour": 14,
        "minute": 30,
        "second": 0,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone": 5.5,
        "user_name": "Aarav"
    }
}

try:
    resp = requests.post(url, json=payload, timeout=10)
    print("Status:", resp.status_code)
    print("Response JSON:")
    data = resp.json()
    print("Answer snippet:", data.get("answer", "")[:300])
    print("Intent:", data.get("intent"))
    print("Answer Source:", data.get("answer_source"))
    print("Related Questions:", data.get("related_questions"))
except Exception as e:
    print("Error:", e)
