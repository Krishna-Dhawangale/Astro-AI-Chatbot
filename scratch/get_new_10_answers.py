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
    "1. What was my exact past life?",
    "2. Who was I in my previous birth?",
    "3. Give me the exact winning lottery numbers.",
    "4. Tell me the exact date I will die.",
    "5. Tell me exactly what will happen to me tomorrow at 3 PM.",
    "6. What will be the exact stock price of a company next month?",
    "7. Can you guarantee that I will become a millionaire?",
    "8. Tell me the exact name of my future spouse.",
    "9. Tell me exactly how many children I will have.",
    "10. Can you predict the exact events that will happen every day for the next year?"
]

async def main():
    out_file = os.path.abspath("scratch/new_10_answers.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, q in enumerate(questions, 1):
            req = ChatRequest(
                query=q,
                user_id=f"test_new_u_{idx}",
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
            ans = ""
            mode = ""
            source = ""
            if hasattr(res, 'body'):
                body_json = json.loads(res.body.decode('utf-8'))
                ans = body_json.get('answer', '')
                mode = body_json.get('answer_mode', '')
                source = body_json.get('answer_source', body_json.get('source', ''))
            elif hasattr(res, 'body_iterator'):
                chunks = []
                async for chunk in res.body_iterator:
                    if isinstance(chunk, bytes):
                        chunks.append(chunk.decode('utf-8'))
                    else:
                        chunks.append(str(chunk))
                ans = "".join(chunks)
                mode = "LLM_ASSISTED"
                source = "EVIDENCE_GROUNDED_LLM"

            f.write(f"### Question {idx}\n")
            f.write(f"**User Prompt**: `{q}`\n")
            f.write(f"**Answer Mode**: `{mode}` | **Answer Source**: `{source}`\n")
            f.write(f"**Chatbot Response**:\n{ans}\n\n")
            f.write("="*60 + "\n\n")

if __name__ == "__main__":
    asyncio.run(main())
