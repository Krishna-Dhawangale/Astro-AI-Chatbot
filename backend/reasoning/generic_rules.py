def apply_generic_rules(evidence: list[str]) -> list[str]:
    return list(evidence)


def analyze_generic_rules(chart_data=None) -> dict:
    return {
        "career": {"matched_rules": []},
        "marriage": {"matched_rules": []},
        "finance": {"matched_rules": []},
        "education": {"matched_rules": []},
        "property": {"matched_rules": []},
        "health": {"matched_rules": []},
    }