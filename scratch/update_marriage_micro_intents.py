import json

new_templates = {
    "marriage_timing": "Namaste {user_name}! Relationship milestones and marriage activation for your {lagna} Ascendant operate through your 7th house in {house_7_sign} (lord: {house_7_lord}) and natural relationship karaka Venus. Under your active {current_dasha}, marriage timing activates when transiting Jupiter or Venus casts a benefic aspect upon your 7th house or 7th lord, initiating favorable alignment for formal engagement and wedding ceremonies.",

    "marriage_probability": "Namaste {user_name}! Analyzing the Vivah Yoga in your {lagna} Ascendant birth chart, the probability of marriage is governed by your 7th house of union ({house_7_sign}, lord: {house_7_lord}) and the strength of Venus/Jupiter. Your horoscope shows positive planetary promise for marriage rather than lifelong denial. With supportive Dasha cycles under {current_dasha}, obstacles dissolve once your 7th lord is strengthened and favorable planetary transits activate your partnership axis.",

    "marriage_decision": "Namaste {user_name}! In your {lagna} Ascendant chart, the decision regarding marriage and lifelong partnership is guided by your 1st house of personal autonomy ({lagna_lord}) and 7th house of union ({house_7_sign}, lord: {house_7_lord}). Your horoscope favors entering marriage rather than remaining single, as a supportive union provides emotional grounding and karmic maturity under {current_dasha}. Choosing a partner with intellectual alignment and emotional patience ensures your personal independence thrives alongside marital harmony.",

    "finance_probability": "Hello {user_name}! Wealth prospects and the probability of achieving financial prosperity in your {lagna} chart are anchored by your 2nd house of wealth ({house_2_sign}, lord: {house_2_lord}) and 11th house of gains ({house_11_sign}, lord: {house_11_lord}). Under {current_dasha}, the likelihood of steady financial accumulation is strong, provided you build disciplined assets and avoid speculative risks.",

    "health_probability": "Namaste {user_name}! Physical resilience and wellness probability for your {lagna} Lagna are governed by your 1st house ({lagna_lord}) and 6th house ({house_6_sign}, lord: {house_6_lord}). Your chart indicates high recuperative stamina; maintaining disciplined routines and mindful stress management shields you from chronic vulnerabilities under {current_dasha}."
}

for path in ['qa_database.json', 'backend/qa_database.json']:
    with open(path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    for k, v in new_templates.items():
        db[k] = v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"Updated {path}: total keys = {len(db)}")
