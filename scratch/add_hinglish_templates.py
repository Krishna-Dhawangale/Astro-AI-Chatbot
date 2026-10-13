import json

hinglish_templates = {
    # CAREER
    "career_timing_hinglish": "Namaste {user_name}! Aapke {lagna} Lagna ke liye professional transitions aur promotions aapke 10th house ({house_10_sign}, lord: {house_10_lord}) aur active {current_dasha} se govern hote hain. Jab transiting Jupiter ya 10th lord aapke 10th ya 11th house par auspicious drishti dalte hain, tab favorable Antardasha sub-periods ke dauran nayi job opportunities aur promotion ke strong yog bante hain.",
    "career_probability_hinglish": "Namaste {user_name}! Aapke {lagna} chart mein career success aur stability ki sambhavna aapke Lagna lord {lagna_lord} aur 10th house lord {house_10_lord} par depend karti hai. Active {current_dasha} ke dauran professional targets achieve karne ke asar kafi strong hain. Structured execution banaye rakhein aur transitional phases mein impulsive decisions se bachein.",
    "career_domain_fit_hinglish": "Namaste {user_name}! Aapke {lagna} chart ke according aapka professional temperament Lagna lord {lagna_lord} aur 10th house ({house_10_sign}) se shape hota hai. Aapke chart ke mutabiq {work_environment} kafi favorable hai. Corporate ya salaried job 6th house stability provide karti hai, jabki business tab flourish karta hai jab 7th aur 11th house ko benefic planetary support mile.",

    # FINANCE
    "finance_timing_hinglish": "Namaste {user_name}! Dhan aagman aur financial milestones primarily 2nd lord ({house_2_lord}) ya 11th lord ({house_11_lord}) ke sub-periods mein activate hote hain. Active {current_dasha} ke dauran jab transiting Jupiter 11th house ({house_11_sign}) ko illuminate karega ya 2nd house ({house_2_sign}) par drishti dalega, tab cash inflow aur financial expansion ka supportive samay banega.",
    "finance_risk_hinglish": "Namaste {user_name}! Aapke {lagna} chart mein financial risk aur debt exposure 6th house ({house_6_sign}, lord: {house_6_lord}) aur 8th house se regulate hote hain. Active {current_dasha} ke dauran speculative market bets (stock market/crypto) ya bina planning ke bada loan lene se bachein. Capital preservation aur secure investments ko priority dein.",
    "wealth_accumulation_hinglish": "Namaste {user_name}! Long-term wealth accumulation aapke 2nd house of assets ({house_2_sign}, lord: {house_2_lord}) aur 11th house of gains ({house_11_sign}, lord: {house_11_lord}) par depend karta hai. Aapke {lagna} Lagna ke liye {financial_prospects}. Disciplined systematic savings aur real assets mein investment aapki lifelong net worth ko steadily badhayega.",
    "finance_probability_hinglish": "Namaste {user_name}! Dhanwaan banne aur financial stability achieve karne ke yog aapke 2nd house ({house_2_sign}, lord: {house_2_lord}) aur 11th house ({house_11_sign}, lord: {house_11_lord}) mein bane Dhana Yogas par depend karte hain. Active {current_dasha} ke under aapke financial accumulation ke asar positive hain, bas high-risk speculation se bachein.",

    # HEALTH
    "health_timing_hinglish": "Namaste {user_name}! Health restoration aur vitality resurgence aapke 1st house (vitality) aur 6th house ({house_6_sign}, lord: {house_6_lord}) se regulate hote hain. Current {current_dasha} ke dauran jaise hi transit align hoga aur Lagna lord {lagna_lord} ko benefic support milega, aapki physical energy aur recovery steadily rebound karegi.",
    "health_mitigation_hinglish": "Namaste {user_name}! Physical vitality aur emotional calm ko protect karne ke liye Lagna lord {lagna_lord} ko strengthen karein aur 6th lord {house_6_lord} ko propitiate karein. Aapke {moon_sign} Moon ke liye daily morning Pranayama, hydration, aur Maha Mrityunjaya Mantra ka jaap chronic stress aur seasonal fatigue ko neutralize karta hai.",
    "health_probability_hinglish": "Namaste {user_name}! Aapke {lagna} Lagna ke liye physical resilience aur wellness probability 1st house ({lagna_lord}) aur 6th house ({house_6_sign}, lord: {house_6_lord}) par depend karti hai. Aapka chart high recuperative stamina show karta hai; disciplined daily routine aur mindful stress management se aap active {current_dasha} mein fit rahenge.",

    # MARRIAGE
    "marriage_timing_hinglish": "Namaste {user_name}! Relationship milestones aur shaadi ki timing aapke 7th house ({house_7_sign}, lord: {house_7_lord}) aur natural karaka Venus se regulate hoti hai. Active {current_dasha} ke dauran jab transiting Jupiter ya Venus aapke 7th house ya 7th lord par shubh drishti dalenge, tab shaadi ke yog activate honge aur wedding ceremonies ke liye favorable window banegi.",
    "marriage_probability_hinglish": "Namaste {user_name}! Aapke {lagna} Ascendant chart mein shaadi hone ke asar (Vivah Yoga) 7th house ({house_7_sign}, lord: {house_7_lord}) aur Venus/Jupiter ki strength se judge hote hain. Aapki horoscope mein lifelong denial ke koi indicators nahi hain; chart shaadi ke high positive probability ko support karta hai. Active {current_dasha} ke dauran temporary delay testing phase hai jo sahi transit aane par clear ho jayega.",
    "marriage_decision_hinglish": "Namaste {user_name}! Aapke {lagna} Lagna chart mein shaadi karne ya na karne ka decision 1st house (personal autonomy, {lagna_lord}) aur 7th house (partnership, {house_7_sign}, lord: {house_7_lord}) par depend karta hai. Aapka horoscope single rehne ke bajaye shaadi ko favor karta hai, kyunki ek supportive partner aapko emotional grounding aur maturity dega under {current_dasha}. Sahi understanding aur matching wavelength ke saath aage badhna kafi beneficial rahega.",
    "partner_profile_hinglish": "Namaste {user_name}! Aapki Vedic chart mein 7th house of marriage {house_7_sign} mein hai, jiske lord {house_7_lord} hain. Aapke life partner {partner_traits} swabhav ke honge. Unka practical aur supportive nature aapke {lagna} Lagna ko deeply ground aur balance karega.",
    "relationship_compatibility_hinglish": "Namaste {user_name}! Vedic astrology mein relationship compatibility Ashtakoota harmony aur 7th house ({house_7_sign}, lord: {house_7_lord}) alignment par depend karti hai. Dono partners ke beech mutual respect aur supportive Moon/Venus placements hone par ek enduring aur intellectually satisfying bond banta hai.",
    "relationship_mitigation_hinglish": "Namaste {user_name}! Shaadi mein aane wale delays aur friction ko dur karne ke liye 7th lord {house_7_lord} ki pooja karein. Mindful communication practice karein, Thursdays ko yellow flowers offer karein, aur Venus/Jupiter ko propitiate karein taaki commitment delays naturally resolve ho sakein.",

    # EDUCATION
    "education_field_fit_hinglish": "Namaste {user_name}! Aapka academic intellect 5th house ({house_5_sign}, lord: {house_5_lord}) aur 9th house of higher wisdom ({house_9_sign}) se govern hota hai. Yeh cosmic alignment analytical, research-oriented, aur strategic specialization ko strongly support karta hai jahan deep focus aur creative problem-solving required ho.",
    "exam_timing_success_hinglish": "Namaste {user_name}! Competitive exams aur interviews mein success 6th house ({house_6_sign}, lord: {house_6_lord}) aur 5th lord {house_5_lord} ke sub-periods mein activate hoti hai. Active {current_dasha} ke dauran Mercury aur Jupiter horas mein focused study aur revision optimal retention aur competitive edge deta hai.",
    "study_abroad_hinglish": "Namaste {user_name}! Overseas education tab indicate hoti hai jab 5th house ka connection 9th house ({house_9_sign}) ya 12th house ({house_12_sign}) se bane. Active {current_dasha} ke dauran {house_9_lord} aur {house_12_lord} ka shubh connection foreign university admissions aur scholarship opportunities create karta hai.",

    # TRAVEL
    "foreign_travel_timing_hinglish": "Namaste {user_name}! International travel aur distant journeys 9th lord ({house_9_lord}) aur 12th lord ({house_12_lord}) ke periods mein trigger hote hain. Active {current_dasha} ke under foreign travel aapke career aur personal perspectives ke liye kafi fruitful aur auspicious rahegi.",
    "foreign_settlement_fit_hinglish": "Namaste {user_name}! Permanent foreign residency aapke 12th house ({house_12_sign}, lord: {house_12_lord}) se govern hoti hai. Jab {house_12_lord} 4th aur 9th house ke saath strong Kendra ya Trikona sambhandh banata hai, tab foreign country mein permanent roots establish karna highly supported hota hai.",
    "travel_obstacle_mitigation_hinglish": "Namaste {user_name}! Visa delays aur relocation hurdles ko clear karne ke liye Rahu ko propitiate karein aur Wednesdays/Saturdays ko Goddess Durga ya Lord Bhairava ki pooja karein. Rahu Beej Mantra ka jaap aur sesame seeds donate karna bureaucratic obstacles ko smooth karta hai.",

    # SPIRITUALITY
    "ishta_devata_guidance_hinglish": "Namaste {user_name}! Aapke 5th house of past-life spiritual merit (Purva Punya) {house_5_sign} mein hai, jiske lord {house_5_lord} hain. Aapki soul ka spiritual compass {ishta_devata} se align hota hai. Daily meditation aur unka sacred mantra jaap inner peace aur mental clarity deta hai.",
    "karmic_dosha_mitigation_hinglish": "Namaste {user_name}! Kundali ke karmic debts aur doshas ko dissolve karne ke liye Saturdays ko charitable acts karein (birds ya cows ko feed karna) aur Maha Mrityunjaya Mantra chant karein. Lagna lord {lagna_lord} ko fortify karne se malefic transit impacts neutralize hote hain.",
    "spiritual_growth_dharma_hinglish": "Namaste {user_name}! Dharma aur spiritual evolution ka 9th house {house_9_sign} mein hai, ruled by {house_9_lord}. Aapke {lagna} Lagna ke liye spiritual progress ethical integrity, authentic guidance, aur daily sacred mindfulness ke through naturally unfold hoti hai.",

    # GEMSTONES
    "gemstone_recommendation_hinglish": "Namaste {user_name}! Aapke {lagna} Lagna ke liye core vitality aur confidence Lagna lord {lagna_lord} se govern hota hai. Aapka prime auspicious gemstone {gemstone_lagna} hai. Isko designated finger par gold ya silver mein dharan karne se immunity aur personal charisma boost hota hai.",
    "gemstone_wearing_rules_hinglish": "Namaste {user_name}! Vedic rules ke mutabiq gemstone natural, unheated hona chahiye aur Shukla Paksha mein planet ke weekday par sunrise time par energize karna chahiye. Gemstone ko raw milk aur Ganga jal se purify karein, seed mantra 108 baar chant karein aur auspicious Hora mein pehnein."
}

for path in ['qa_database.json', 'backend/qa_database.json']:
    with open(path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    for k, v in hinglish_templates.items():
        db[k] = v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"Updated {path}: total keys = {len(db)}")
