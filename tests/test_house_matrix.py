import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.reasoning.lordship import get_house_details
from backend.router.semantic_parser import analyze_query_intent
from backend.router.capability_gate import evaluate_capability, ExecutionPath, ordinal_house
from backend.reasoning.llm_renderer import build_compact_evidence_package

chart = {
    'ascendant': {'rashi': 'Aries'},
    'planets': {
        'Sun': {'house': 1, 'rashi': 'Aries'},
        'Moon': {'house': 2, 'rashi': 'Taurus'},
        'Mars': {'house': 1, 'rashi': 'Aries'},
        'Mercury': {'house': 2, 'rashi': 'Taurus'},
        'Jupiter': {'house': 3, 'rashi': 'Gemini'},
        'Venus': {'house': 4, 'rashi': 'Cancer'},
        'Saturn': {'house': 5, 'rashi': 'Leo'},
        'Rahu': {'house': 6, 'rashi': 'Virgo'},
        'Ketu': {'house': 12, 'rashi': 'Pisces'}
    }
}
dasha = {'current_mahadasha': 'Rahu', 'current_antardasha': 'Moon'}

print(f"| {'House':<6} | {'Sign':<12} | {'Lord':<10} | {'Placement':<14} | {'Occupants':<18} | {'Direct Fact Route':<20} | {'LLM Grounding':<14} |")
print("|" + "-"*8 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*16 + "|" + "-"*20 + "|" + "-"*22 + "|" + "-"*16 + "|")

for h in range(1, 13):
    det = get_house_details(chart, h)
    q_sign = f"Which sign is in my {ordinal_house(h)} house?"
    prof_sign = analyze_query_intent(q_sign)
    dec_sign = evaluate_capability(prof_sign, chart)
    
    q_interp = f"What does my {ordinal_house(h)} house indicate?"
    pkg = build_compact_evidence_package('general', 'general', chart, [], dasha, question=q_interp)
    llm_ok = (f"{h}H" in pkg['verified_facts'] and f"{h}L" in pkg['verified_facts'] and f"{h}H_occupants" in pkg['verified_facts'])
    
    occ_str = ', '.join(det['occupants']) if det['occupants'] else 'None (empty)'
    place_str = f"{det['lord']}@{det['lord_house']}H"
    route_str = dec_sign.path.value
    status_str = "PASS" if (dec_sign.path == ExecutionPath.LOCAL_DIRECT_FACT and llm_ok) else "FAIL"
    
    print(f"| {h:<6} | {det['sign']:^12} | {det['lord']:^10} | {place_str:^14} | {occ_str:^18} | {route_str:^20} | {status_str:^14} |")
