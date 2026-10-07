import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))
logging.disable(logging.CRITICAL)

from backend.main import _handle_chat_response, ChatRequest

questions = [
    "What was my exact past life?",
    "Who was I in my previous birth?",
    "Give me the exact winning lottery numbers.",
    "Tell me the exact date I will die.",
    "Tell me exactly what will happen to me tomorrow at 3 PM.",
    "What will be the exact stock price of a company next month?",
    "Can you guarantee that I will become a millionaire?",
    "Tell me the exact name of my future spouse.",
    "Tell me exactly how many children I will have.",
    "Can you predict the exact events that will happen every day for the next year?"
]

async def main():
    out_file = os.path.abspath("scratch/clean_10_answers.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, q in enumerate(questions, 1):
            req = ChatRequest(
                query=q,
                user_id=f"test_user_q{idx}",
                name="Rahul",
                birth_year=1995,
                birth_month=5,
                birth_day=15,
                birth_hour=14.5,
                latitude=28.6139,
                longitude=77.2090,
                timezone=5.5
            )
            res = await _handle_chat_response(req)
            if hasattr(res, 'body'):
                body_json = json.loads(res.body.decode('utf-8'))
                ans = body_json.get('answer', '')
                mode = body_json.get('answer_mode', '')
                source = body_json.get('source', '')
                f.write(f"### Question {idx}: {q}\n")
                f.write(f"**Answer Mode**: `{mode}` | **Source**: `{source}`\n")
                f.write(f"**Chatbot Output**:\n{ans}\n\n")
                f.write("="*60 + "\n\n")

if __name__ == "__main__":
    asyncio.run(main())
