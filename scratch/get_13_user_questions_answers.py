"""
Script: scratch/get_13_user_questions_answers.py
Executes the user's 13 domain questions against backend/main.py and outputs clean results.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from backend.main import _handle_chat_response, ChatRequest

questions = [
    "1. How does my current Mahadasha affect my career?",
    "2. How does my current Antardasha influence my life?",
    "3. What does my chart indicate about my career growth?",
    "4. What does my chart indicate about marriage timing?",
    "5. What does my chart indicate about financial stability?",
    "6. What does my chart indicate about higher education?",
    "7. What does my chart indicate about property matters?",
    "8. What does my chart indicate about relationships?",
    "9. What are the strongest career indicators in my chart?",
    "10. What are the strongest marriage indicators in my chart?",
    "11. Which planets are most important for my career?",
    "12. Which planets influence my 7th house?",
    "13. How does my 10th lord influence my career?"
]

async def run_questions():
    output_lines = []
    
    for i, q_raw in enumerate(questions, 1):
        q = q_raw.split(". ", 1)[1]
        req = ChatRequest(
            query=q,
            user_id=f"user_eval_13_{i}",
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

    with open("scratch/13_user_questions_answers.txt", "w", encoding="utf-8") as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    asyncio.run(run_questions())
