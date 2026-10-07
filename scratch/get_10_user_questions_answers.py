"""
Script: scratch/get_10_user_questions_answers.py
Executes the user's 10 specific questions against backend/main.py and outputs clean results.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from backend.main import _handle_chat_response, ChatRequest

questions = [
    "1. Can you explain my overall career situation from my chart?",
    "2. What are the major themes of my life right now?",
    "3. Can you give me a short interpretation of my current astrological period?",
    "4. How do my different chart placements work together?",
    "5. What are the strongest themes you see in my birth chart?",
    "6. Can you explain my chart in simple language?",
    "7. What kind of career path appears most aligned with my chart?",
    "8. How do my personality and career indicators connect?",
    "9. What major changes might this period represent according to my chart?",
    "10. Can you summarize what my current Dasha means for me?"
]

async def run_questions():
    output_lines = []
    
    for i, q_raw in enumerate(questions, 1):
        q = q_raw.split(". ", 1)[1]
        req = ChatRequest(
            query=q,
            user_id=f"user_eval_{i}",
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
        ans_text = ""
        ans_mode = "UNKNOWN"
        ans_src = "UNKNOWN"
        gemini_calls = 0

        if hasattr(res, 'body'):
            body_json = json.loads(res.body.decode('utf-8'))
            ans_text = body_json.get("answer", "")
            ans_mode = body_json.get("answer_mode", "LOCAL")
            ans_src = body_json.get("answer_source", "LOCAL")
            gemini_calls = body_json.get("gemini_calls", 0)
        elif hasattr(res, 'body_iterator'):
            chunks = []
            async for chunk in res.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode('utf-8'))
            raw = "".join(chunks)

            # Strip follow up line if present
            if "\n[FOLLOW_UPS]:" in raw:
                raw = raw.split("\n[FOLLOW_UPS]:")[0]
            if raw.endswith(" (LLM)"):
                raw = raw[:-6]
            ans_text = raw.strip()
            ans_mode = "LLM_ASSISTED"
            ans_src = "GEMINI_GROUNDED"
            gemini_calls = 1

        # Clean source label from text if wrapped
        clean_ans = ans_text
        if clean_ans.startswith("[") and "]" in clean_ans[:30]:
            clean_ans = clean_ans.split("]", 1)[1].strip()

        block = (
            f"### Question {i}\n"
            f"**User Question**: `{q}`\n"
            f"**Answer Mode**: `{ans_mode}` | **Answer Source**: `{ans_src}` | **Gemini Calls**: `{gemini_calls}`\n"
            f"**Chatbot Response**:\n"
            f"{clean_ans}\n\n"
            f"{'='*60}\n"
        )
        print(block)
        output_lines.append(block)

    with open("scratch/10_user_questions_answers.txt", "w", encoding="utf-8") as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    asyncio.run(run_questions())
