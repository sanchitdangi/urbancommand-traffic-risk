"""Real-time weather fetcher using wttr.in (no API key required).
Results are cached per city for 15 minutes to avoid rate limiting."""

import requests
import time

_cache = {}       # { city_name: { "weather": str, "ts": float } }
CACHE_TTL = 900   # 15 minutes

# wttr.in weather code → our internal label
def _code_to_label(code: int) -> str:
    if code in [113, 116]:
        return "Clear"
    if code in [119, 122]:
        return "Cloudy"
    if code == 248:
        return "Fog"          # Dense fog only
    if code in [143, 260]:   # Mist / Freezing fog → treat as hazy/cloudy, not fog
        return "Cloudy"
    if code in [305, 308, 356, 359, 389]:
        return "Heavy Rain"
    if code in [176, 263, 266, 281, 284, 293, 296, 299,
                302, 311, 317, 353, 362, 386]:
        return "Rain"
    if code in [200]:
        return "Thunderstorm"
    if code in [323, 326, 329, 332, 335, 338,
                368, 371, 374, 377, 392, 395]:
        return "Snow / Sleet"
    return "Clear"


# Cities the API can resolve (exclude highway corridors / abstract zones)
# Months where dense fog is climatologically valid in North India
_FOG_VALID_MONTHS = {10, 11, 12, 1, 2}

# Locations that are not valid cities for the weather API
_UNSUPPORTED = {"Unknown", "Abstract", "Highway", "Rural Road", "Sector-X"}

def get_real_weather(city_name: str) -> str | None:
    """
    Return real current weather label for a city.
    Returns None if city is unsupported or the API call fails.
    Post-validates Fog against season to avoid impossible combinations.
    """
    if not city_name or any(u in city_name for u in _UNSUPPORTED):
        return None

    now = time.time()
    cached = _cache.get(city_name)
    if cached and now - cached["ts"] < CACHE_TTL:
        label = cached["weather"]
        return _seasonal_guard(label)

    try:
        url  = f"https://wttr.in/{city_name}?format=j1"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            data    = resp.json()
            code    = int(data["current_condition"][0]["weatherCode"])
            label   = _code_to_label(code)
            _cache[city_name] = {"weather": label, "ts": now}
            return _seasonal_guard(label)
    except Exception:
        pass

    return None


from datetime import datetime as _dt

def _seasonal_guard(label: str) -> str:
    """Downgrade Fog to Cloudy if we're outside the North India fog season."""
    if label == "Fog" and _dt.now().month not in _FOG_VALID_MONTHS:
        return "Cloudy"   # May–Sep: no dense fog in Indian plains
    return label
