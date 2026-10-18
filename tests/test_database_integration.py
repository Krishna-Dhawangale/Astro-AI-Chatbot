import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import (
    init_db,
    log_chat_interaction,
    get_unresolved_queries,
    mark_query_resolved,
    AsyncSessionLocal,
    User,
    ChatSession,
    ChatMessage,
)
from sqlalchemy import select


def test_database_init_and_direct_logging():
    import asyncio

    async def _run():
        await init_db()

        # Log a standard LOCAL turn
        msg1 = await log_chat_interaction(
            session_id="pytest-session-001",
            user_query="Which house represents career in Vedic astrology?",
            bot_response="The 10th house (Karma Bhava) represents career and profession.",
            execution_mode="LOCAL",
            detected_intents=["career_inquiry"],
            input_tokens=12,
            output_tokens=48,
            calculation_footer="Vedic standard astrology rule",
            is_unresolved=False,
            user_id="pytest-user-001",
            user_profile={"name": "AstroTester", "birth_year": 1995, "birth_month": 5, "birth_day": 20},
        )
        assert msg1 is not None
        assert msg1.execution_mode == "LOCAL"
        assert msg1.is_unresolved is False
        assert msg1.input_tokens == 12
        assert msg1.output_tokens == 48

        # Log an UNRESOLVED failed turn
        msg2 = await log_chat_interaction(
            session_id="pytest-session-001",
            user_query="Can you tell me the stock price of Apple next week?",
            bot_response="Sorry, I cannot answer stock market predictive questions.",
            execution_mode="UNRESOLVED",
            detected_intents=["unsupported_financial"],
            input_tokens=10,
            output_tokens=15,
            calculation_footer="Out of domain boundary",
            is_unresolved=True,
            user_id="pytest-user-001",
        )
        assert msg2 is not None
        assert msg2.is_unresolved is True
        assert msg2.execution_mode == "UNRESOLVED"

        # Verify query directly from PostgreSQL
        async with AsyncSessionLocal() as session:
            unresolved_records, count = await get_unresolved_queries(session, limit=10)
            assert count >= 1
            matching = [r for r in unresolved_records if r.id == msg2.id]
            assert len(matching) == 1
            assert matching[0].user_query == "Can you tell me the stock price of Apple next week?"

    asyncio.run(_run())


def test_fastapi_endpoints_integration():
    with TestClient(app) as client:
        # 1. Ask a question that succeeds (educational FAQ)
        chat_payload = {
            "user_id": "api-test-session-999",
            "query": "What does Jupiter signify in Vedic astrology?",
            "name": "IntegrationTester"
        }
        res = client.post("/chat", json=chat_payload)
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data

        # 2. Trigger an unhandled / empty query that must be flagged as unresolved
        empty_payload = {
            "user_id": "api-test-session-999",
            "query": "   ",
        }
        res_empty = client.post("/chat", json=empty_payload)
        assert res_empty.status_code == 400

        # 3. Query the admin endpoint to ensure the unresolved query is captured
        admin_res = client.get("/admin/unresolved-queries?limit=20")
        assert admin_res.status_code == 200
        admin_json = admin_res.json()
        assert admin_json["status"] == "success"
        assert admin_json["total_unresolved"] >= 1
        
        # Verify fields in records
        found_records = admin_json["unresolved_queries"]
        assert len(found_records) > 0
        first = found_records[0]
        assert "id" in first
        assert "session_id" in first
        assert "user_query" in first
        assert "bot_response" in first
        assert "execution_mode" in first
        assert "is_unresolved" in first
        assert first["is_unresolved"] is True
