import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print("=== Testing FastAPI Chatbot Diagnostic & Stability ===")
    
    test_cases = [
        # 1. Basic Career Query
        {
            "endpoint": "/chat",
            "payload": {
                "user_id": "test_user_career",
                "query": "career",
                "name": "Arjun",
                "birth_year": 1995,
                "birth_month": 5,
                "birth_day": 12,
                "birth_hour": 14.5
            }
        },
        # 2. Basic Dasha Query (User specifically reported crash here)
        {
            "endpoint": "/chat",
            "payload": {
                "user_id": "test_user_dasha",
                "query": "dasha",
                "name": "Priya",
                "birth_year": 1998,
                "birth_month": 8,
                "birth_day": 20,
                "birth_hour": 9.0
            }
        },
        # 3. Direct Dasha Inquiries
        {
            "endpoint": "/chat",
            "payload": {
                "user_id": "test_user_dasha_tell",
                "query": "tell me about my dasha",
                "name": "Rahul"
            }
        },
        # 4. Uninitialized Session / Disconnected User (Zero birth details)
        {
            "endpoint": "/chat",
            "payload": {
                "user_id": "brand_new_uninitialized_user",
                "query": "what is my career outlook?"
            }
        },
        # 5. /query Endpoint (Alternative main endpoint)
        {
            "endpoint": "/query",
            "payload": {
                "user_id": "test_query_endpoint_user",
                "query": "will I get a promotion?"
            }
        },
        # 6. Hinglish Marriage Query
        {
            "endpoint": "/chat",
            "payload": {
                "user_id": "test_hinglish_user",
                "query": "meri shadi kab hogi"
            }
        },
        # 7. /api/chat with ApiChatRequest schema
        {
            "endpoint": "/api/chat",
            "payload": {
                "prompt": "career guidance",
                "user_profile": {
                    "name": "Sunita",
                    "age": 28
                }
            }
        }
    ]

    all_passed = True
    for idx, tc in enumerate(test_cases, 1):
        url = f"{BASE_URL}{tc['endpoint']}"
        print(f"\n[{idx}] Testing {tc['endpoint']} with query: '{tc['payload'].get('query') or tc['payload'].get('prompt')}'")
        try:
            resp = requests.post(url, json=tc["payload"], timeout=15)
            print(f"Status Code: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"FAILED: Expected 200, got {resp.status_code}. Response: {resp.text[:200]}")
                all_passed = False
                continue

            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                content = resp.json()
                ans = content.get("answer", "")
                ans_mode = content.get("answer_mode", "")
                ans_src = content.get("answer_source", "")
                print(f"Type: JSON | Mode: {ans_mode} | Source: {ans_src}")
            else:
                ans = resp.text
                print(f"Type: STREAMING (text/plain)")
            
            # Check for forbidden crash string
            if "Sorry, I couldn't generate a response right now" in ans:
                print("FAILED: Global server crash error returned!")
                all_passed = False
                continue

            print(f"Answer Preview:\n{ans[:160]}...")
            print("PASSED")

        except Exception as e:
            print(f"EXCEPTION during test: {e}")
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("ALL TESTS PASSED! Chatbot server is rock-solid and crash-free.")
    else:
        print("SOME TESTS FAILED.")

if __name__ == "__main__":
    test_endpoints()
