import asyncio
import json
import httpx

async def test_live_api():
    base_url = "http://127.0.0.1:8000/api/chat"

    test_queries = [
        # 1. Multi-domain English
        {
            "query": "Can you analyze my career, finance, health, and recommended gemstones all at once?",
            "expected_lang": "English",
        },
        # 2. Multi-domain Hinglish
        {
            "query": "Mera career, finance, health aur recommended gemstones kaisa rahega?",
            "expected_lang": "Hinglish",
        },
        # 3. Direct fact English
        {
            "query": "What is my moon sign?",
            "expected_lang": "English",
        },
        # 4. Direct fact Hinglish
        {
            "query": "Mera moon sign kya hai?",
            "expected_lang": "Hinglish",
        },
    ]

    print("=" * 70)
    print("TESTING LIVE SERVER /api/chat ENDPOINT FOR LANGUAGE MIRRORING")
    print("=" * 70 + "\n")

    async with httpx.AsyncClient(timeout=45.0) as client:
        for t in test_queries:
            payload = {
                "prompt": t["query"],
                "user_profile": {
                    "name": "Alex",
                    "birth_year": 1995,
                    "birth_month": 5,
                    "birth_day": 15,
                    "birth_hour": 14.5,
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                    "timezone": 5.5
                }
            }

            print(f"Sending Query ({t['expected_lang']}): `{t['query']}`")
            resp = await client.post(base_url, json=payload)
            print(f"Status Code: {resp.status_code}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

            text = resp.text
            # If JSON response (e.g. Mode 1 direct fact)
            if resp.headers.get("content-type", "").startswith("application/json"):
                data = resp.json()
                clean_text = data.get("answer", "")
            else:
                # Streaming response
                clean_text = text.split("\n[FOLLOW_UPS]:")[0].strip()

            print(f"Response Received:\n{clean_text}\n")

            if t["expected_lang"] == "English":
                assert not any(hw in clean_text.lower() for hw in ["aapka", "aapke", "hai", "karein", "anusaar"]), (
                    f"Expected English but found Hindi words in response: {clean_text}"
                )
                print(">>> Verified: Responded in English.")
            else:
                has_hi = any(hw in clean_text.lower() for hw in ["aap", "aapka", "aapke", "hai", "karein", "anusaar", "rahega", "aur", "ki"])
                assert has_hi, f"Expected Hinglish but found no Hindi words in response: {clean_text}"
                print(">>> Verified: Responded in Hinglish.")
            print("-" * 70 + "\n")

    print("ALL LIVE API LANGUAGE MIRRORING TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_live_api())
