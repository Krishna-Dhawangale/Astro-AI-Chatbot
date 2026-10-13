import asyncio
import json
import httpx


async def test_live_api_chat():
    payload = {
        "prompt": "Can you analyze my career, finance, health, and recommended gemstones all at once?",
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

    print("POSTING to http://127.0.0.1:8000/api/chat ...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post("http://127.0.0.1:8000/api/chat", json=payload)
        print(f"Status Code: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('content-type')}")
        text = resp.text
        print("\n--- Response Body ---")
        print(text)
        print("---------------------")

        assert "[Local Database + LLM -" in text, "Missing hybrid badge!"
        assert "**Astrological Insight:**" in text
        assert "**Actionable Advice:**" in text
        print("\n[SUCCESS] /api/chat correctly executed Hybrid Multi-Domain RAG!")


if __name__ == "__main__":
    asyncio.run(test_live_api_chat())
