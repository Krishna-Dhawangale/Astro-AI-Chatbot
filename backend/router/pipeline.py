from .domain import predict_domain
from .complexity import predict_complexity
from .intent import predict_intent


def route_question(question: str):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    # --------------------------------------------------------
    # 1. DOMAIN
    # --------------------------------------------------------

    domain = predict_domain(question)

    # --------------------------------------------------------
    # 2. COMPLEXITY
    # --------------------------------------------------------

    complexity = predict_complexity(question)

    # --------------------------------------------------------
    # 3. INTENT
    # --------------------------------------------------------

    intent = predict_intent(question)

    # --------------------------------------------------------
    # 4. ROUTING DECISION
    # --------------------------------------------------------

    if complexity == "simple":
        route = "faq"

    elif complexity == "needs_chart":
        route = "astrology"

    else:
        route = "unknown"

    return {
        "question": question,
        "domain": domain,
        "complexity": complexity,
        "intent": intent,
        "route": route
    }