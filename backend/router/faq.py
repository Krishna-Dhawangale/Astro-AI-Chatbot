FAQ_RESPONSES = {
    "what is vedic astrology":
        "Vedic astrology is a traditional system that interprets planetary positions using a birth chart.",

    "what is a birth chart":
        "A birth chart maps the planets and houses for the time and place of birth.",
}


def find_faq_answer(question: str) -> str | None:
    normalized = " ".join(question.lower().split()).rstrip("?")
    return FAQ_RESPONSES.get(normalized)