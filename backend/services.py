import os
import requests
from dotenv import load_dotenv

load_dotenv()

FREE_ASTROLOGY_KEY = os.getenv("FREE_ASTROLOGY_API_KEY") or os.getenv("FREE_ASTROLOGY_KEY")
PROKERALA_CLIENT_ID = os.getenv("PROKERALA_CLIENT_ID")
PROKERALA_CLIENT_SECRET = os.getenv("PROKERALA_CLIENT_SECRET")

# 1. Free Astrology API - Planetary Positions
def fetch_planet_positions(year, month, day, hour, minute, second, lat, lon, tz=5.5):
    url = "https://json.freeastrologyapi.com/planets/extended"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": FREE_ASTROLOGY_KEY or ""
    }
    payload = {
        "year": year,
        "month": month,
        "date": day,
        "hours": hour,
        "minutes": minute,
        "seconds": second,
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "config": {
            "observation_point": "topocentric",
            "ayanamsha": "lahiri"
        }
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e), "statusCode": 500}


# 2. Free Astrology API - Vimshottari Dasha
def fetch_dasha_details(year, month, day, hour, minute, second, lat, lon, tz=5.5):
    url = "https://json.freeastrologyapi.com/vimsottari/maha-dasas-and-antar-dasas"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": FREE_ASTROLOGY_KEY or ""
    }
    payload = {
        "year": year,
        "month": month,
        "date": day,
        "hours": hour,
        "minutes": minute,
        "seconds": second,
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "config": {
            "observation_point": "topocentric",
            "ayanamsha": "lahiri"
        }
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e), "statusCode": 500}


# 3. Prokerala API - OAuth Token & SVG Chart
def fetch_prokerala_chart(year, month, day, hour, minute, lat, lon, tz_offset="+05:30"):
    # Step A: Get Access Token
    token_url = "https://api.prokerala.com/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": PROKERALA_CLIENT_ID,
        "client_secret": PROKERALA_CLIENT_SECRET
    }
    token_res = requests.post(token_url, data=token_data)
    access_token = token_res.json().get("access_token")

    # Step B: Request Chart SVG
    chart_url = "https://api.prokerala.com/v2/astrology/chart"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "image/svg+xml"
    }
    
    formatted_dt = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00{tz_offset}"
    params = {
        "chart_style": "north-indian",
        "chart_type": "rasi",
        "coordinates": f"{lat},{lon}",
        "datetime": formatted_dt,
        "ayanamsa": 1,
        "la": "en"
    }
    
    chart_res = requests.get(chart_url, headers=headers, params=params)
    return chart_res.text  # Returns SVG XML String