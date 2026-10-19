"""
backend/astrology/house_engine.py
=================================
Vedic Astrology House (Bhava) Engine & Significator Analysis
-----------------------------------------------------------
Implements deep Vedic astrology interpretations for all 12 Bhavas (houses):
1. Sanskrit Bhava designations (Tanu, Dhana, Sahaja, Sukha, Putra, Ari, Kalatra,
   Ayur, Bhagya, Karma, Labha, Vyaya).
2. Natural Kalapurusha Rashis & Natural Planetary Karakas (Bhava Karakas).
3. Natal Cusp Sign and Ruling Lord determination based on user's Ascendant.
4. House Occupants (planets positioned in the house) and Vedic interpretations.
5. Accurate, authentic Vedic synthesis for queries like:
   - "What does the 7th house mean?"
   - "Planets in 10th house"
   - "What planets are in my 4th house?"
   - "Tell me about my 7th house"
6. Full bilingual support: Natural Conversational Hinglish & English.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

ZODIAC_RULERS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

# ---------------------------------------------------------------------------
# 12 Bhavas: Complete Vedic Astrological Knowledge Base
# ---------------------------------------------------------------------------
BHAVA_KNOWLEDGE_BASE: Dict[int, Dict[str, Any]] = {
    1: {
        "sanskrit_name": "Tanu Bhava (Lagna)",
        "english_title": "1st House of Self, Physical Body & Vitality",
        "kalapurusha_sign": "Aries (Mesha)",
        "karakas": ["Sun (Surya)"],
        "governs_en": "Self-identity, physical appearance, longevity, general health, constitution, temperament, life outlook, and personal vitality.",
        "governs_hi": "Aapka astitva, vyaktitva (personality), swasthya, shareerik banawat, aatmavishwas, aur jeevan ki disha.",
        "vedic_summary_en": "The 1st House (Lagna / Tanu Bhava) is the anchor of the entire birth chart. Ruled by the Sun as natural karaka, it represents the soul incarnating into physical reality. A strong 1st house confers high vitality, leadership qualities, and resilience against life's obstacles.",
        "vedic_summary_hi": "Lagna (Tanu Bhava) poori kundali ka aadhar stambh hai. Surya iske natural karaka hain. Yeh aapke aatmavishwas, swasthya, vyaktitva aur jeevan mein safalta paane ki kshamta ko darshata hai.",
        "planet_meanings": {
            "Sun": {"en": "Radiant confidence, strong leadership instincts, robust vitality, and authoritative presence.", "hi": "Prabhavshali vyaktitva, leadership kshamta, aur mazboot aatmavishwas pradan karta hai."},
            "Moon": {"en": "Empathetic, intuitive, adaptable, and emotionally expressive personality with a magnetic charm.", "hi": "Sensitve, aakarshak aur bhavuk vyaktitva deta hai."},
            "Mars": {"en": "Dynamic energy, courageous disposition, direct communicative approach, and competitive spirit.", "hi": "Utsahi, sahasik, energetic aur thoda tez krodh wala swabhav banata hai."},
            "Mercury": {"en": "Sharp intellect, youthful demeanor, analytical communication, and exceptional learning agility.", "hi": "Tez dimaag, baat karne mein nipun, aur business minded banata hai."},
            "Jupiter": {"en": "Wisdom, dignity, generous spirit, spiritual inclination, and natural philosophical guidance.", "hi": "Gyan, shant swabhav, aadar-samman, aur dharmik pravritti deta hai."},
            "Venus": {"en": "Charming presence, artistic sensibilities, graceful demeanor, and magnetic social appeal.", "hi": "Sundar, aakarshak vyaktitva, kala aur soundarya mein ruchi deta hai."},
            "Saturn": {"en": "Mature, disciplined, patient, hardworking nature, with success built through perseverance.", "hi": "Gambhir, anushasit, mehanti aur dhairya se kaam lene wala banata hai."},
            "Rahu": {"en": "Unconventional charisma, ambitious drive, strong desires to innovate, and worldly focus.", "hi": "Mahatvakankshi, innovative aur duniya se alag sochne wala banata hai."},
            "Ketu": {"en": "Introspective, detached, spiritual inclinations, and intuitive self-awareness.", "hi": "Aadhyatmik, shant, aur materialistic cheezon se thoda detached banata hai."},
        }
    },
    2: {
        "sanskrit_name": "Dhana Bhava & Kutumba Bhava",
        "english_title": "2nd House of Accumulated Wealth, Family & Speech",
        "kalapurusha_sign": "Taurus (Vrishabha)",
        "karakas": ["Jupiter (Guru) for Wealth", "Mercury (Budha) for Speech"],
        "governs_en": "Accumulated savings, liquid wealth, family lineage, ancestral roots, speech tone, dietary habits, and moral values.",
        "governs_hi": "Sanchit dhan (savings), parivar, vaani (speech), khana-peena, aur sanskar.",
        "vedic_summary_en": "The 2nd House governs wealth preservation, family upbringing, and the power of the spoken word. Jupiter acts as the primary Karaka for financial assets, while Mercury influences eloquence and truthfulness in communication.",
        "vedic_summary_hi": "2nd house aapke bank balance, parivarik rishton aur baat karne ki shaili ko darshata hai. Jupiter dhan ke aur Mercury vaani ke karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Authoritative speech, proud family heritage, financial stability through government or executive roles.", "hi": "Prabhavshali vaani aur parivar mein sthiti ko mazboot banata hai."},
            "Moon": {"en": "Fluctuating financial flow, gentle and soothing speech, deep emotional connection to family.", "hi": "Dhan ke aavagman mein utar-chadhav aur madhur vaani deta hai."},
            "Mars": {"en": "Direct, blunt speech, assertive wealth-building drive, potential minor family disputes over assets.", "hi": "Spasht vaani aur dhan kamane ke liye aggressive mehnat deta hai."},
            "Mercury": {"en": "Clever commercial instincts, persuasive speech, wealth through advisory, trading, or accounting.", "hi": "Meethi vaani, commercial acumen, aur trading se dhan labh deta hai."},
            "Jupiter": {"en": "Abundant financial growth, virtuous speech, respected family reputation, and generous nature.", "hi": "Shubh dhan yog, gyanvaan vaani aur parivarik samriddhi deta hai."},
            "Venus": {"en": "Sweet speech, love for luxury items, financial prosperity through beauty, arts, or hospitality.", "hi": "Madhur bol-chaal, luxury aur aish-o-aaram se dhan labh deta hai."},
            "Saturn": {"en": "Frugal habits, financial growth through steady accumulation over time, conservative speech.", "hi": "Dhan sanchay mein dhairya aur dheere-dheere sthir samriddhi deta hai."},
            "Rahu": {"en": "Unconventional income streams, foreign earnings, ambitious appetite for material resources.", "hi": "Foreign sources ya sudden means se dhan kamane ke avenues banata hai."},
            "Ketu": {"en": "Detached attitude toward material wealth, spiritual speech, simple living values.", "hi": "Dhan ke prati thoda santosh aur aadhyatmik drishtikon deta hai."},
        }
    },
    3: {
        "sanskrit_name": "Sahaja Bhava & Parakrama Bhava",
        "english_title": "3rd House of Courage, Siblings & Self-Effort",
        "kalapurusha_sign": "Gemini (Mithuna)",
        "karakas": ["Mars (Mangal) for Courage/Siblings", "Mercury (Budha) for Communication"],
        "governs_en": "Courage, valor, initiative, younger siblings, short journeys, writing, communication, hands/arms, and hobbies.",
        "governs_hi": "Sahas, parakram, chhote bhai-behan, self-effort, writing, communication, aur chhoti yatrayein.",
        "vedic_summary_en": "The 3rd House is an Upachaya (growth) house representing will power and initiative. Mars provides fighting spirit while Mercury supports media, writing, and digital communication skills.",
        "vedic_summary_hi": "3rd house aapke sahas, mehnat aur bhai-behno ke sath rishton ko darshata hai. Mars parakram aur Mercury communication ke karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Brave heart, decisive actions, respected standing among peers, independent initiative.", "hi": "Nidar swabhav aur khud ke bal-boote par aage badhne ki shakti deta hai."},
            "Moon": {"en": "Imaginative writing, frequent short travels, gentle interaction with siblings.", "hi": "Rachanatmak soch aur yatrao se seekhne ka avsar deta hai."},
            "Mars": {"en": "Exceptional courage, athletic or technical prowess, proactive drive, strong competitive edge.", "hi": "Jabardast sahas aur mushkilon se ladne ki taqat deta hai."},
            "Mercury": {"en": "Gift of expression, skilled in networking, journalism, digital content, or sales.", "hi": "Behtareen communication skills aur business networking kshamta deta hai."},
            "Jupiter": {"en": "Inspirational counseling, ethical communication, supportive sibling bonds.", "hi": "Gyanvardhak baat-cheet aur bhai-behno ka sahyog deta hai."},
            "Venus": {"en": "Artistic hands, creative hobbies (music, design, photography), pleasant social communication.", "hi": "Kala, sangeet aur creative fields mein ruchi deta hai."},
            "Saturn": {"en": "Methodical perseverance, patient handiwork, practical long-term efforts yielding lasting results.", "hi": "Lagaatar mehnat aur kathin parishram se safalta deta hai."},
            "Rahu": {"en": "Daring risk-taker, mastery of digital media, bold unconventional initiatives.", "hi": "Risk lene ki kshamta aur digital media mein safalta deta hai."},
            "Ketu": {"en": "Quiet internal courage, skilled in solitary crafts, spiritual writing.", "hi": "Shant sahas aur deep analytical/spiritual research deta hai."},
        }
    },
    4: {
        "sanskrit_name": "Sukha Bhava & Matru Bhava",
        "english_title": "4th House of Home, Mother, Property & Peace of Mind",
        "kalapurusha_sign": "Cancer (Karka)",
        "karakas": ["Moon (Chandra) for Mother & Mind", "Venus (Shukra) for Vehicles", "Mars (Mangal) for Real Estate"],
        "governs_en": "Mother, domestic happiness, inner peace, home environment, real estate, land, vehicles (Vahana), foundational education, and emotional security.",
        "governs_hi": "Mata, ghar ka sukh, mansik shanti, property, zameen-makan, gadi (vahana), aur prathmik shiksha.",
        "vedic_summary_en": "The 4th House is a prime Kendra (quadrant) house representing the foundation of emotional life. Moon rules inner tranquility, Venus governs vehicle comforts, and Mars governs landed property.",
        "vedic_summary_hi": "4th house aapke ghar, mata ke sukh, property aur mansik sukoon ka kendra hai. Moon mata aur shanti ke, Venus vahan ke, aur Mars property ke karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Strong domestic leadership, noble family heritage, ownership of solid structural property.", "hi": "Ghar mein prabhav aur acchi property ka sukh deta hai."},
            "Moon": {"en": "Deep maternal bond, peaceful domestic life near water or nature, emotional sensitivity.", "hi": "Mata ke sath gehra sneh aur mansik shanti pradan karta hai."},
            "Mars": {"en": "Real estate acquisition, passion for land development, need for harmony in the home.", "hi": "Bhoomi, makan aur real estate se labh deta hai."},
            "Mercury": {"en": "Scholarly home environment, educational achievements, lively family discussions.", "hi": "Acchi shiksha aur ghar mein intellectual mahol deta hai."},
            "Jupiter": {"en": "Expansive home happiness, virtuous mother, spacious residence, deep inner fulfillment.", "hi": "Ghar mein sukh-shanti, bado ka aashirwad aur vishal makan ka sukh deta hai."},
            "Venus": {"en": "Beautiful residence, luxury cars/vehicles, artistic interior decor, affectionate domestic ties.", "hi": "Sundar ghar, luxury gadi aur aish-o-aaram ka sukh deta hai."},
            "Saturn": {"en": "Late domestic stability, ancient or durable real estate, practical sense of responsibility.", "hi": "Dheere-dheere sthir property aur parivarik zimmedari nibhata hai."},
            "Rahu": {"en": "Residence away from birthplace or abroad, modern high-tech home, sudden vehicle upgrades.", "hi": "Birthplace se door rehne ya modern ghar/gadi ka yog banata hai."},
            "Ketu": {"en": "Spiritual atmosphere at home, seeking peace through solitude, detachment from materialistic comfort.", "hi": "Ghar mein shanti aur aadhyatmik vatavaran pasand karta hai."},
        }
    },
    5: {
        "sanskrit_name": "Putra Bhava & Purva Punya Bhava",
        "english_title": "5th House of Intelligence, Children, Romance & Past Merit",
        "kalapurusha_sign": "Leo (Simha)",
        "karakas": ["Jupiter (Guru) for Children & Wisdom"],
        "governs_en": "Children (progeny), creative intelligence (Buddhi), past-life karmic merit (Purva Punya), higher education, romance, speculative investments, and mantra siddhi.",
        "governs_hi": "Santaan (children), buddhi (intellect), purva punya, romance, share market / speculation, aur gyan.",
        "vedic_summary_en": "The 5th House is a supreme Trikona (trine) house representing Lakshmi sthana (fortune and divine merit). Jupiter signifies progeny and spiritual wisdom, while solar energy empowers creative self-expression.",
        "vedic_summary_hi": "5th house trikona bhav hai jo gyan, buddhi, santaan aur purva janm ke punyo ko darshata hai. Jupiter iske mukhya karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Sharp intellect, dignified creativity, honorable children, talent in governance and administration.", "hi": "Tez buddhi, aadarsh santaan aur administration mein safalta deta hai."},
            "Moon": {"en": "Creative imagination, emotional tenderness toward children, strong romantic inclinations.", "hi": "Creative soch, bhavuk prem sambandh aur kalatmak ruchi deta hai."},
            "Mars": {"en": "Analytical mind, passion for sports, engineering or logic, dynamic decision-making.", "hi": "Technical buddhi, competitive spirit aur quick decision-making deta hai."},
            "Mercury": {"en": "Genius analytical intellect, flair for writing, math, finance, and witty communication.", "hi": "Tez analytical dimag, stock market aur calculation mein nipun banata hai."},
            "Jupiter": {"en": "Profound wisdom, virtuous children, academic brilliance, natural spiritual inclinations.", "hi": "Uchha vidya, santaan sukh, aur pavitra gyan deta hai."},
            "Venus": {"en": "Romantic fulfillment, artistic genius, charming children, success in entertainment or design.", "hi": "Romance, kala, fashion aur khushnuma prem sambandh deta hai."},
            "Saturn": {"en": "Deep philosophical concentration, serious studies, delayed but durable speculative returns.", "hi": "Gehri padhai, dhairya aur long-term learning mein safalta deta hai."},
            "Rahu": {"en": "Unconventional intellect, speculative windfalls in tech or markets, out-of-the-box creativity.", "hi": "Share market aur creative fields mein sudden gains ka yog banata hai."},
            "Ketu": {"en": "Intuitive spiritual insight, mastery over ancient mantras, detached approach to speculation.", "hi": "Intuition, dhyan aur mantra vidya mein ruchi deta hai."},
        }
    },
    6: {
        "sanskrit_name": "Ari Bhava & Roga-Ripu-Rina Bhava",
        "english_title": "6th House of Health, Daily Service, Debts & Competition",
        "kalapurusha_sign": "Virgo (Kanya)",
        "karakas": ["Mars (Mangal) for Overcoming Enemies", "Saturn (Shani) for Service & Discipline"],
        "governs_en": "Daily work routines, service, immune resilience, ailments, overcoming competitors, debts (Rina), legal disputes, and maternal uncles.",
        "governs_hi": "Daily service / job, swasthya (immunity), shatru (competitors), karz (loans), aur competitive exams.",
        "vedic_summary_en": "The 6th House is an Upachaya (house of growth) and a Dusthana (challenging house). While it points to obstacles, malefic planets here (Mars, Saturn, Sun) grant victory over rivals and success in competitive exams.",
        "vedic_summary_hi": "6th house mehnat, competitive exams, aur mushkilon par vijay paane ka bhav hai. Mars aur Saturn jaise graha yahan competitors ko harane ki shakti dete hain.",
        "planet_meanings": {
            "Sun": {"en": "Shatru Hanta (crushing competition), high immunity, leadership in public service or medicine.", "hi": "Competitors par vijay, mazboot immunity aur sarkari naukri ke liye anukool banata hai."},
            "Moon": {"en": "Sensitivity to digestive health, service-oriented caregiving nature, mood affected by routine.", "hi": "Health aur diet ka dhyan rakhna zaroori hota hai, caring nature deta hai."},
            "Mars": {"en": "Unstoppable competitive drive, excels in sports, law, or surgical fields, overcomes adversaries.", "hi": "Shatruo par jeet, competitive exams mein safalta aur zabardast fighting spirit deta hai."},
            "Mercury": {"en": "Analytical problem-solver, success in auditing, law, data science, and service management.", "hi": "Problems solve karne mein tez dimaag aur service sector mein safalta deta hai."},
            "Jupiter": {"en": "Diplomatic conflict resolution, protection against serious illness, benevolent service.", "hi": "Shanti se vivado ko suljhana aur swasthya ki raksha karta hai."},
            "Venus": {"en": "Diplomatic handling of colleagues, needs attention to lifestyle balance and kidney/sugar health.", "hi": "Colleagues ke sath acche sambandh aur balanced lifestyle ki zaroorat hoti hai."},
            "Saturn": {"en": "Iron resilience, destroys opposition, enduring discipline, success through persistent service.", "hi": "Shandar discipline, shatruo ka naash, aur long-term job stability deta hai."},
            "Rahu": {"en": "Overcomes rivals through tactical thinking, excels in complex corporate or legal environments.", "hi": "Corporate ya legal challenges mein tactical jeet dilata hai."},
            "Ketu": {"en": "Indifference to rivals, interest in alternative healing and holistic therapies.", "hi": "Holistic healing aur swasthya ke prati natural awareness deta hai."},
        }
    },
    7: {
        "sanskrit_name": "Kalatra Bhava & Yuvati Bhava",
        "english_title": "7th House of Marriage, Spouse, Partnerships & Contracts",
        "kalapurusha_sign": "Libra (Tula)",
        "karakas": ["Venus (Shukra) for Relationships/Love", "Jupiter (Guru) for Husband in Female Chart"],
        "governs_en": "Spouse, marriage timing, marital harmony, business partners, public relations, trade, legal contracts, and overseas journeys.",
        "governs_hi": "Vivah (marriage), jeevan saathi (spouse), business partnership, public relations, aur trade contracts.",
        "vedic_summary_en": "The 7th House directly faces the Lagna, reflecting the significant other and public interactions. Venus is the primary Karaka of marriage and mutual affection. Its strength indicates relationship harmony, spouse's character, and business alliance success.",
        "vedic_summary_hi": "7th house vivah, jeevan saathi aur business partnership ka kendra hai. Venus iske mukhya karaka hain. Yeh house rishto mein prem, samjhauta aur mutual growth ko darshata hai.",
        "planet_meanings": {
            "Sun": {"en": "Dignified, proud partner; partner may hold executive or administrative standing; requires mutual respect.", "hi": "Atma-sammami aur authoritative jeevan saathi; mutual respect banaye rakhna zaroori hota hai."},
            "Moon": {"en": "Emotional, nurturing, attractive spouse; harmonious relationship fueled by intuitive understanding.", "hi": "Sundar, bhavuk aur dekhbhal karne wala jeevan saathi deta hai."},
            "Mars": {"en": "Dynamic, energetic, ambitious spouse (Manglik factor); passionate relationship requiring patience.", "hi": "Urjavan aur ambitious jeevan saathi; rishte mein thoda dhairya aur samajh zaroori hoti hai."},
            "Mercury": {"en": "Witty, intelligent, communicative partner; intellectual friendship forms the base of marriage.", "hi": "Buddhiman, baat-cheet mein nipun aur dost jaisa jeevan saathi deta hai."},
            "Jupiter": {"en": "Virtuous, wise, prosperous, and supportive partner; high marital bliss and social respect.", "hi": "Gyanvaan, shant aur sammanit jeevan saathi; vivah mein shubhata aur sthirta deta hai."},
            "Venus": {"en": "Charming, loving, aesthetically refined spouse; romantic compatibility and marital pleasure.", "hi": "Sundar, prem karne wala aur kalatmak jeevan saathi; sukhad dampatya jeevan deta hai."},
            "Saturn": {"en": "Mature, loyal, dutiful, dependable partner; late marriage often brings greater stability.", "hi": "Samajhdar, vafadar aur zimmedar partner; thoda der se vivah sthirta lata hai."},
            "Rahu": {"en": "Unconventional or inter-cultural partner; dynamic and progressive marital dynamics.", "hi": "Progressive, modern ya alag background wala jeevan saathi deta hai."},
            "Ketu": {"en": "Spiritually inclined partner; relationship emphasizes soul connection over superficial bonds.", "hi": "Aadhyatmik soch wala aur deep understanding rakhne wala partner deta hai."},
        }
    },
    8: {
        "sanskrit_name": "Ayur Bhava & Randhra Bhava",
        "english_title": "8th House of Longevity, Transformation, Secrets & Occult",
        "kalapurusha_sign": "Scorpio (Vrishchika)",
        "karakas": ["Saturn (Shani) for Longevity"],
        "governs_en": "Longevity (Ayushya), sudden transformations, unearned wealth (inheritance, insurance, taxes), occult sciences, astrology, psychological depth, and hidden matters.",
        "governs_hi": "Aayu (longevity), sudden parivartan, virasat (inheritance), astrology/occult, aur psychological depth.",
        "vedic_summary_en": "The 8th House represents profound transformation, the mysteries of life and death, and hidden assets. Saturn acts as the natural Karaka of longevity. While challenging, strong placements here grant profound intuitive research skills and sudden breakthroughs.",
        "vedic_summary_hi": "8th house aayu, sudden changes, virasat aur astrology/occult sciences ka bhav hai. Saturn iske karaka hain. Yeh house deep research aur transformation deta hai.",
        "planet_meanings": {
            "Sun": {"en": "Interest in ancient wisdom, deep internal resilience, transformative life phases.", "hi": "Ghan gyan mein ruchi aur jeevan ke kathin samay mein aatmik shakti deta hai."},
            "Moon": {"en": "Deep psychic intuition, emotional metamorphosis, interest in mystical realms.", "hi": "Gehri intuition aur mansik parivartan ki kshamta deta hai."},
            "Mars": {"en": "Fearless exploration of hidden truths, interest in surgery, investigative forensics, or engineering.", "hi": "Nidar swabhav aur hidden problems ko investigate karne ki taqat deta hai."},
            "Mercury": {"en": "Exceptional investigative mind, talent for data cryptography, forensic audit, or esoteric study.", "hi": "Deep research, data analysis aur astrology mein tez dimaag deta hai."},
            "Jupiter": {"en": "Protection against life crises, profound occult knowledge, unearned financial gains or inheritance.", "hi": "Musibato se raksha, virasat se labh aur gupt gyan mein kripa deta hai."},
            "Venus": {"en": "Inherited wealth, secretive romantic matters, peaceful recovery from financial changes.", "hi": "Virasat ya partner ke finance se labh deta hai."},
            "Saturn": {"en": "Long lifespan (Deergha Ayu), patient enduring through life trials, interest in deep history.", "hi": "Lambi aayu, dhairya aur deep research mein safalta deta hai."},
            "Rahu": {"en": "Sudden unexpected financial windfalls, interest in occult, deep psychological insights.", "hi": "Achanak dhan labh ya sudden events ke zariye parivartan deta hai."},
            "Ketu": {"en": "Moksha orientation, natural adept in astrology, mysticism, and spiritual transcendence.", "hi": "Astrology, kundali aur aadhyatmik gyan mein natural talent deta hai."},
        }
    },
    9: {
        "sanskrit_name": "Bhagya Bhava & Dharma Bhava",
        "english_title": "9th House of Fortune, Dharma, Higher Wisdom & Father",
        "kalapurusha_sign": "Sagittarius (Dhanu)",
        "karakas": ["Jupiter (Guru) for Dharma & Wisdom", "Sun (Surya) for Father"],
        "governs_en": "Luck (Bhagya), moral character, father, gurus/mentors, higher academic pursuits, long-distance journeys, spirituality, and divine grace.",
        "governs_hi": "Bhagya (luck), dharma, pita, guru, higher education, tirth yatra, aur eeshwariya kripa.",
        "vedic_summary_en": "The 9th House is the most auspicious Trikona (trine) house, known as Bhagya Sthana. Jupiter rules higher wisdom and ethics, while the Sun signifies paternal lineage. A strong 9th house unlocks smooth fortune and spiritual elevation.",
        "vedic_summary_hi": "9th house kundali ka sabse shubh trikona bhav hai jise Bhagya Sthana kehte hain. Jupiter gyan aur dharma ke, aur Surya pita ke karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Dignified father, honorable reputation, alignment with righteous dharma and truth.", "hi": "Pita ka aashirwad, sammanit jeevan aur bhagya ka sath deta hai."},
            "Moon": {"en": "Spiritual devotion, frequent travels to sacred places, gentle moral values.", "hi": "Dharmik yatrayein aur pavitra vichaar pradan karta hai."},
            "Mars": {"en": "Passionate defender of justice, energetic crusader for righteousness, adventurous world traveler.", "hi": "Nyaya ke liye ladne wala aur sahasik yatrao mein ruchi lene wala banata hai."},
            "Mercury": {"en": "Scholar of philosophies, gifted educator, author of wisdom literature, global publisher.", "hi": "Higher education, teaching aur philosophical writing mein safalta deta hai."},
            "Jupiter": {"en": "Supreme Bhagya Yoga, divine blessings, renowned spiritual mentor, flourishing higher education.", "hi": "Prabal bhagya yog, guruo ka margdarshan aur ucch shiksha deta hai."},
            "Venus": {"en": "Love for sacred art and music, fortunate marriage, harmonious cultural travels.", "hi": "Kala, dharmik sangeet aur videsh yatrao se prasannata deta hai."},
            "Saturn": {"en": "Traditional religious discipline, earned philosophical depth through rigorous study.", "hi": "Dharmik anushasan aur sthir bhagyodaya deta hai."},
            "Rahu": {"en": "Cross-cultural wisdom, global travel, progressive spiritual beliefs.", "hi": "Videsh yatra aur modern spiritual drishtikon deta hai."},
            "Ketu": {"en": "Direct spiritual liberation instinct, pilgrim spirit, deep detachment from dogmas.", "hi": "Moksha marg, tirth yatra aur deep spiritual realization deta hai."},
        }
    },
    10: {
        "sanskrit_name": "Karma Bhava & Rajyashrita Bhava",
        "english_title": "10th House of Career, Profession, Authority & Public Status",
        "kalapurusha_sign": "Capricorn (Makara)",
        "karakas": ["Sun (Authority)", "Saturn (Perseverance/Work)", "Mercury (Business)", "Jupiter (Counsel/Ethics)"],
        "governs_en": "Career path, professional success, public reputation, status, honors, executive authority, government relations, and life achievements.",
        "governs_hi": "Career, naukri, business, public status, pad-pratishtha, authority, aur karma.",
        "vedic_summary_en": "The 10th House is the highest Kendra (quadrant) house representing midday culmination and public impact. Ruled by Saturn and the Sun as natural karakas, it maps professional achievements, administrative power, and societal contributions.",
        "vedic_summary_hi": "10th house Karma Bhava hai jo aapke career, professional safalta, samaaj mein izzat aur authority ko darshata hai. Surya aur Shani iske mukhya karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Digbala (directional strength)! Executive authority, government prominence, top leadership status.", "hi": "Digbala strength! Sarkari pad, leadership aur sammanit career deta hai."},
            "Moon": {"en": "Public-facing career, strong emotional connect with crowds, adaptable career trajectory.", "hi": "Public relations, HR ya creative fields mein safalta deta hai."},
            "Mars": {"en": "Digbala! Courageous enterprise, top managerial command, excellence in engineering, defense, or surgery.", "hi": "Digbala! Technical leadership, police/defense, ya high executive roles deta hai."},
            "Mercury": {"en": "Commercial brilliance, exceptional managerial communication, success in tech, finance, or media.", "hi": "Business, IT, data aur advisory services mein behtareen career banata hai."},
            "Jupiter": {"en": "Respected leadership, ethical executive authority, success as judge, professor, CEO, or counselor.", "hi": "Sammanit pad, advisor, teacher, ya legal/management leadership deta hai."},
            "Venus": {"en": "Glamour, arts, luxury, media, diplomacy, or hospitality leadership, praised public image.", "hi": "Design, entertainment, luxury brands aur diplomacy mein safalta deta hai."},
            "Saturn": {"en": "Digbala! Ultimate perseverance, long-lasting career legacy, mastery over complex organizations.", "hi": "Stheer safalta, kathin parishram se banna bada career aur shandar authority deta hai."},
            "Rahu": {"en": "Unconventional career heights, sudden meteoric rise in modern tech, media, or politics.", "hi": "Modern technologies, global business aur tez career growth deta hai."},
            "Ketu": {"en": "Expert specialist behind the scenes, seeks purpose and fulfillment over mere title.", "hi": "Subject matter expert, behind-the-scenes guidance aur meaningful kaam deta hai."},
        }
    },
    11: {
        "sanskrit_name": "Labha Bhava & Aya Bhava",
        "english_title": "11th House of Gains, Income, Social Network & Aspirations",
        "kalapurusha_sign": "Aquarius (Kumbha)",
        "karakas": ["Jupiter (Guru) for Expansion & Inflow"],
        "governs_en": "Income streams, monetary gains, elder siblings, social network circles, fulfillment of ambitions, and large-scale organizational alliances.",
        "governs_hi": "Aamdani (gains), labh, bade bhai-behan, social networks, ichha-poorti (wishes fulfillment).",
        "vedic_summary_en": "The 11th House is the supreme Labha Sthana (house of profit and realization). Every planet placed here tends to deliver positive material yield over time. Jupiter serves as the primary Karaka of abundance and wish fulfillment.",
        "vedic_summary_hi": "11th house Labha Bhava hai jo income flow aur manokamanaon ki poorti ko darshata hai. Is house mein lagbhag sabhi graha shubh aur dhan-labh pradayak hote hain.",
        "planet_meanings": {
            "Sun": {"en": "Profits through influential contacts, high-level associations, reliable monetary income.", "hi": "Bade logo se sampark aur steady financial gains deta hai."},
            "Moon": {"en": "Gains through public commerce, supportive female mentors and expansive social alliances.", "hi": "Social connections aur public dealing se accha dhan labh deta hai."},
            "Mars": {"en": "Aggressive financial gains, profits through enterprise, engineering, and competitive ventures.", "hi": "Vyapar aur proactive efforts se bada financial profit deta hai."},
            "Mercury": {"en": "Multiple income streams, financial gains through trading, contracts, technology, and analytics.", "hi": "Ek se zyada income sources aur commercial networking se labh deta hai."},
            "Jupiter": {"en": "Tremendous wealth expansion, virtuous elder associates, natural fulfillment of major aspirations.", "hi": "Vishal dhan labh, ichhao ki poorti aur shubh sahyog deta hai."},
            "Venus": {"en": "Luxurious gains, prosperity through arts, high-society friends, continuous financial flow.", "hi": "Luxury items, creative projects aur dosto ke sahyog se labh deta hai."},
            "Saturn": {"en": "Steady, permanent accumulation of wealth, dependable long-term institutional network.", "hi": "Sthir aur permanent wealth accumulation deta hai."},
            "Rahu": {"en": "Immense financial windfalls, massive global social network, exponential wealth gains.", "hi": "Zabardast financial gains aur global networks se labh deta hai."},
            "Ketu": {"en": "Selective social circle, contentment with simple finances, gains channeled to charity.", "hi": "Chuninda acche dost aur spiritual santosh deta hai."},
        }
    },
    12: {
        "sanskrit_name": "Vyaya Bhava & Moksha Bhava",
        "english_title": "12th House of Foreign Lands, Solitude, Sleep & Liberation",
        "kalapurusha_sign": "Pisces (Meena)",
        "karakas": ["Saturn (Renunciation)", "Ketu (Moksha/Liberation)", "Venus (Bed Comforts/Sleep)"],
        "governs_en": "Foreign travel/settlement, spiritual liberation (Moksha), restorative sleep (Shayana Sukha), subconscious mind, expenditures, ashrams, and hospitals.",
        "governs_hi": "Videsh yatra/settlement, moksha, neend (restorative sleep), kharch (expenses), aur subconscious mind.",
        "vedic_summary_en": "The 12th House is the final house of the zodiac, governing dissolution, overseas journeys, and ultimate spiritual liberation (Moksha). Ketu acts as the Karaka for Moksha, while Venus signifies peaceful bed comforts.",
        "vedic_summary_hi": "12th house videsh yatra, overseas settlement, achhi neend aur aatmik shanti ka bhav hai. Ketu moksha ke aur Venus shayan sukh ke karaka hain.",
        "planet_meanings": {
            "Sun": {"en": "Foreign career success, quiet spiritual dignity, expenditures aligned with noble charitable causes.", "hi": "Videsh mein safalta aur dharmik/charitable kaamo par kharch deta hai."},
            "Moon": {"en": "Vivid dreams, overseas travel across water bodies, need for calming bedtime routines.", "hi": "Videsh yatra aur calming sleep hygiene ki zaroorat deta hai."},
            "Mars": {"en": "Energy directed overseas, active subconscious, benefits from channelized physical exertion before sleep.", "hi": "Foreign projects mein energetic role aur regular physical exercise se shanti deta hai."},
            "Mercury": {"en": "International business analysis, foreign language agility, research conducted in solitude.", "hi": "Foreign languages aur cross-border analysis mein safalta deta hai."},
            "Jupiter": {"en": "Spiritual enlightenment, generous philanthropic expenses, safe foreign residence.", "hi": "Aadhyatmik vikas, daan-punya aur videsh mein samman deta hai."},
            "Venus": {"en": "Supreme bed comforts (Shayana Sukha), love for luxury resorts/retreats, artistic peace in foreign lands.", "hi": "Behtareen shayan sukh, luxury travel aur shanti deta hai."},
            "Saturn": {"en": "Permanent foreign settlement, deep meditative discipline, structured detachment.", "hi": "Permanent overseas settlement aur meditative discipline deta hai."},
            "Rahu": {"en": "Relocation to distant foreign cultures, unconventional international ventures, night-owl habits.", "hi": "Videsh mein settlement aur international opportunities open karta hai."},
            "Ketu": {"en": "Moksha Karaka! High spiritual potential, deep meditation mastery, intuitive transcendence.", "hi": "Moksha Karaka! Gehri dhyan sadhana aur aatmik gyan deta hai."},
        }
    }
}


def extract_house_number(query: str) -> Optional[int]:
    """
    Extracts house number (1 through 12) from various English, Hinglish, and numeric patterns.
    """
    if not query:
        return None
    q = query.lower().strip()

    # Numeric patterns: "7th house", "10th house", "1st house", "house 7", "house 10"
    num_match = re.search(r"\b(?:house|bhava?|ghar)\s*#?\s*([1-9]|1[0-2])\b", q)
    if num_match:
        return int(num_match.group(1))

    ord_match = re.search(r"\b([1-9]|1[0-2])\s*(?:st|nd|rd|th)?\s*(?:house|bhava?|ghar|cusp)\b", q)
    if ord_match:
        return int(ord_match.group(1))

    # Ordinal words map
    word_map = {
        "first": 1, "pehla": 1, "pehli": 1, "pehle": 1, "lagna": 1, "1st": 1,
        "second": 2, "dusra": 2, "doosra": 2, "dusri": 2, "2nd": 2,
        "third": 3, "teesra": 3, "tisra": 3, "teesri": 3, "3rd": 3,
        "fourth": 4, "chautha": 4, "chauthi": 4, "4th": 4,
        "fifth": 5, "panchva": 5, "panchwa": 5, "panchvi": 5, "5th": 5,
        "sixth": 6, "chhattha": 6, "chatha": 6, "chathi": 6, "6th": 6,
        "seventh": 7, "saatva": 7, "satva": 7, "satvi": 7, "7th": 7,
        "eighth": 8, "aathva": 8, "athva": 8, "athvi": 8, "8th": 8,
        "ninth": 9, "nauva": 9, "navam": 9, "navmi": 9, "9th": 9,
        "tenth": 10, "dasva": 10, "daswa": 10, "dasvi": 10, "10th": 10,
        "eleventh": 11, "gyarahva": 11, "gyarva": 11, "11th": 11,
        "twelfth": 12, "barahva": 12, "barva": 12, "12th": 12,
    }

    for word, h_num in word_map.items():
        if re.search(r"\b" + re.escape(word) + r"\s+(?:house|bhava?|ghar)\b", q):
            return h_num

    # Standalone house queries like "planets in 10th", "tell me about 7th"
    for word, h_num in word_map.items():
        if re.search(r"\b(?:in|about|of|my)\s+" + re.escape(word) + r"\b", q):
            return h_num

    return None


def is_house_query(query: str) -> bool:
    """
    Identifies whether the incoming query is focused on astrological houses / bhava karakas.
    """
    if not query:
        return False
    q = query.lower()

    # Explicit house mentions
    if re.search(r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|pehla|dusra|teesra|chautha|panchva|chhattha|satva|athva|navam|dasva|gyarahva|barahva)\s*(?:house|bhava?|ghar)\b", q):
        return True

    # "Planets in X house"
    if re.search(r"\bplanets?\s+(?:in|placed\s+in|sitting\s+in)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)?|my)\s*house\b", q):
        return True

    # Meaning of X house
    if re.search(r"\b(?:what\s+does|meaning\s+of|tell\s+me\s+about|kya\s+hota\s+hai)\s+.*(?:house|bhava)\b", q):
        return True

    # Bhava karaka queries
    if any(k in q for k in ["bhava karaka", "karakas of", "house lord", "house significator", "house signification"]):
        return True

    return False


def get_house_sign_and_lord(asc_sign: str, house_num: int) -> Tuple[str, str]:
    """
    Computes the zodiac sign on a given house cusp and its planetary ruler.
    """
    if asc_sign not in ZODIAC_SIGNS:
        asc_sign = "Aries"
    asc_idx = ZODIAC_SIGNS.index(asc_sign)
    house_sign = ZODIAC_SIGNS[(asc_idx + house_num - 1) % 12]
    lord = ZODIAC_RULERS.get(house_sign, "Mars")
    return house_sign, lord


def get_planets_in_house(chart_data: Dict[str, Any], house_num: int) -> List[str]:
    """
    Extracts all planets currently positioned in the specified house from the normalized chart.
    """
    occupants = []
    if not chart_data or not isinstance(chart_data, dict):
        return occupants

    planets = chart_data.get("planets", {})
    if isinstance(planets, dict):
        for p_name, p_info in planets.items():
            if isinstance(p_info, dict) and p_info.get("house") == house_num:
                occupants.append(p_name)
    return occupants


def synthesize_house_analysis(
    query: str,
    chart_data: Optional[Dict[str, Any]] = None,
    is_hinglish: bool = False,
    asc_sign: Optional[str] = None
) -> Dict[str, Any]:
    """
    Produces a complete, authentic Vedic astrology interpretation for house queries.
    Covers Sanskrit Bhava Karakas, Sign on Cusp, Ruling Lord, and Occupants.
    """
    h_num = extract_house_number(query) or 7  # default to 7 if general query
    house_meta = BHAVA_KNOWLEDGE_BASE.get(h_num, BHAVA_KNOWLEDGE_BASE[7])

    sanskrit_name = house_meta["sanskrit_name"]
    english_title = house_meta["english_title"]
    kalapurusha_sign = house_meta["kalapurusha_sign"]
    karakas = ", ".join(house_meta["karakas"])

    # Determine personalized chart details
    has_chart = bool(chart_data and isinstance(chart_data, dict) and chart_data.get("planets"))
    ascendant_sign = asc_sign
    if not ascendant_sign and chart_data and isinstance(chart_data, dict):
        asc_info = chart_data.get("ascendant", {})
        if isinstance(asc_info, dict):
            ascendant_sign = asc_info.get("rashi") or asc_info.get("sign")

    if not ascendant_sign or ascendant_sign not in ZODIAC_SIGNS:
        ascendant_sign = "Virgo"  # default fallback if uncomputed

    h_sign, h_lord = get_house_sign_and_lord(ascendant_sign, h_num)
    occupants = get_planets_in_house(chart_data or {}, h_num) if has_chart else []

    # Check if asking specifically about occupants
    q_low = query.lower()
    is_occupants_query = any(k in q_low for k in ["planets in", "graha", "occupant", "kaunse graha", "kaun se graha", "sitting in"])

    # Build response
    if is_hinglish:
        # Conversational Hinglish response
        intro = (
            f"Vedic Astrology ke anusaar, **{h_num}th House** ko **{sanskrit_name}** kaha jata hai.\n\n"
            f"**Core Significations (Karakas & Adhikar):**\n"
            f"- **Natural Karakas**: {karakas}\n"
            f"- **Kalapurusha Rashi**: {kalapurusha_sign}\n"
            f"- **Mukhya Kshetra**: {house_meta['governs_hi']}\n\n"
            f"{house_meta['vedic_summary_hi']}\n"
        )

        chart_section = (
            f"\n**Aapki Kundali ({ascendant_sign} Lagna) ke Anusaar:**\n"
            f"- **House Cusp Rashi**: Aapka {h_num}th house **{h_sign}** rashi mein sthit hai.\n"
            f"- **Ruling Lord**: Is bhav ke swami **{h_lord}** hain.\n"
        )

        if occupants:
            occ_list_str = ", ".join(occupants)
            occ_details = []
            for p in occupants:
                p_text = house_meta["planet_meanings"].get(p, {}).get("hi", f"{p} is bhav mein vishesh prabhav dalte hain.")
                occ_details.append(f"  • **{p}**: {p_text}")
            occ_str = "\n".join(occ_details)
            occupants_section = (
                f"- **Sthit Graha (Planets in {h_num}th House)**: **{occ_list_str}**\n"
                f"{occ_str}\n"
            )
        else:
            occupants_section = (
                f"- **Sthit Graha**: Aapke {h_num}th house mein koi natal planet sthit nahi hai. "
                f"Vedic astrology mein is sthiti mein bhav ke fal iske swami **{h_lord}** ki sthiti aur dasha ke anusaar nirdharit hote hain.\n"
            )

        conclusion = (
            f"\n**Vedic Insight:** {h_num}th house ke shubh parinam ke liye apne {h_lord} ko anukool rakhna "
            f"aur sambandhit natural karaka ({karakas.split()[0]}) ke dhyan se jeevan mein balance banaye rakhna labhkari hota hai."
        )

        final_answer = f"{intro}{chart_section}{occupants_section}{conclusion}"

    else:
        # English Response
        intro = (
            f"In Vedic Astrology, the **{h_num}th House** is traditionally known as **{sanskrit_name}** ({english_title}).\n\n"
            f"**Core Astrological Significations:**\n"
            f"- **Natural Karakas (Significators)**: {karakas}\n"
            f"- **Kalapurusha Zodiac Sign**: {kalapurusha_sign}\n"
            f"- **Governs**: {house_meta['governs_en']}\n\n"
            f"{house_meta['vedic_summary_en']}\n"
        )

        chart_section = (
            f"\n**In Your Personal Horoscope ({ascendant_sign} Ascendant):**\n"
            f"- **Zodiac Sign on Cusp**: Your {h_num}th House falls in **{h_sign}**.\n"
            f"- **Ruling Lord**: Governed by **{h_lord}**.\n"
        )

        if occupants:
            occ_list_str = ", ".join(occupants)
            occ_details = []
            for p in occupants:
                p_text = house_meta["planet_meanings"].get(p, {}).get("en", f"Influences the {h_num}th house according to its natural qualities.")
                occ_details.append(f"  • **{p}**: {p_text}")
            occ_str = "\n".join(occ_details)
            occupants_section = (
                f"- **Planetary Occupants**: **{occ_list_str}**\n"
                f"{occ_str}\n"
            )
        else:
            occupants_section = (
                f"- **Planetary Occupants**: No natal planets are posited in your {h_num}th house. "
                f"In Vedic astrology, an unoccupied house is interpreted primarily through its ruling lord (**{h_lord}**) and planetary aspects casting upon this cusp.\n"
            )

        conclusion = (
            f"\n**Vedic Synthesis:** To optimize results for the {h_num}th house, observe the placement of its ruler (**{h_lord}**) "
            f"and strengthen the natural significator ({karakas.split()[0]}) in your daily routines."
        )

        final_answer = f"{intro}{chart_section}{occupants_section}{conclusion}"

    return {
        "answer": final_answer,
        "house_num": h_num,
        "sanskrit_name": sanskrit_name,
        "sign": h_sign,
        "lord": h_lord,
        "karakas": karakas,
        "occupants": occupants,
        "answer_mode": "HOUSE_ANALYSIS",
        "answer_source": "VEDIC_HOUSE_ENGINE",
        "gemini_calls": 0,
        "evidence_complete": True,
    }
