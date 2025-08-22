# day_month_pik.py — clean & strict int IDs
import os, math, re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import requests

WEATHER_KEY = os.environ.get("weather_API_KEY")

# -----------------------
# 시간/날짜 유틸
# -----------------------
def _now_kr() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def _today_kr() -> date:
    return _now_kr().date()

def _month_window_kr() -> Tuple[date, date]:
    today = _today_kr()
    first = today.replace(day=1)
    if first.month == 12:
        next_first = first.replace(year=first.year + 1, month=1, day=1)
    else:
        next_first = first.replace(month=first.month + 1, day=1)
    last = next_first - timedelta(days=1)
    return first, last

def parse_date(s: Any) -> Optional[date]:
    if not s: return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).date()
        except Exception:
            pass
    return None

def to_float(x: Any, default: Optional[float]=None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

# -----------------------
# 텍스트/토크나이즈
# -----------------------
def tokenize(text: str) -> List[str]:
    if not text: return []
    return re.findall(r"[가-힣a-z0-9]+", text.lower())

# -----------------------
# 거리 계산
# -----------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# -----------------------
# 스키마 정규화 + ID 정수 보정
# -----------------------
def _to_int_id(x):
    try:
        return int(x)
    except Exception:
        return None

def norm_post(p: Dict[str, Any]) -> Dict[str, Any]:
    """외부 스키마(예: contents/startDate/viewCount ...)를 내부 공통 키로 통일"""
    pid = _to_int_id(p.get("id") or p.get("postId"))
    return {
        "id":            pid,
        "title":         p.get("title"),
        "content":       p.get("contents") or p.get("content") or "",
        "area":          p.get("area") or "",
        "sigungu":       p.get("sigungu") or "",
        "category":      p.get("category") or p.get("category_name") or "",
        "start_date":    p.get("start_date") or p.get("startDate"),
        "end_date":      p.get("end_date") or p.get("endDate"),
        "view_count":    p.get("view_count") or p.get("viewCount") or 0,
        "like_count":    p.get("like_count") or p.get("likeCount") or 0,
        # 선택/부재 가능 필드
        "place":         p.get("place") or "",
        "placeAddr":     p.get("placeAddr") or p.get("address") or "",
        "gps_x":         p.get("gps_x") or p.get("lng"),
        "gps_y":         p.get("gps_y") or p.get("lat"),
        "avg_rating":    p.get("avg_rating"),
        "review_count":  p.get("review_count") or 0,
        "reviews":       p.get("reviews") or [],
    }

# -----------------------
# 감성/실내외 판별
# -----------------------
EMO_MAP = {
    "조용한": ["조용","차분","한적","고요","잔잔","힐링","명상","서정","잔잔한"],
    "신나는": ["신나","라이브","댄스","밴드","페스티벌","흥겨운","업템포","디제잉","edm","불꽃"],
    "가족":   ["가족","아이","어린이","체험","키즈","유아","보호자","가족형","패밀리"],
    "야외":   ["야외","공원","광장","야외무대","야외공연","플리마켓","거리","야외전시"],
    "실내":   ["실내","전시장","뮤지엄","미술관","박물관","홀","극장","공연장","컨벤션","갤러리","아트센터"],
}
INDOOR_KWS  = set(EMO_MAP["실내"])
OUTDOOR_KWS = set(list(EMO_MAP["야외"]) + ["캠핑","강변","해변","산책","야시장"])

def collect_text(p: Dict[str, Any]) -> str:
    n = norm_post(p)
    parts = [
        str(n["title"] or ""),
        str(n["content"] or ""),
        str(n["place"] or ""),
        str(n["placeAddr"] or ""),
        str(n["area"] or ""),
        str(n["sigungu"] or ""),
        str(n["category"] or ""),
    ]
    for rv in (n["reviews"] or []):
        parts.append(str(rv.get("content") or rv.get("contents") or ""))
    return " ".join(parts).lower()

def detect_inout(blob: str):
    tokens = set(tokenize(blob))
    indoor_hits  = sum(1 for kw in INDOOR_KWS  if kw in blob or kw in tokens)
    outdoor_hits = sum(1 for kw in OUTDOOR_KWS if kw in blob or kw in tokens)
    indoor_score  = min(1.0, indoor_hits/3.0) if indoor_hits  else 0.0
    outdoor_score = min(1.0, outdoor_hits/3.0) if outdoor_hits else 0.0
    if indoor_score > outdoor_score: label = "indoor"
    elif outdoor_score > indoor_score: label = "outdoor"
    else: label = "mixed"
    return indoor_score, outdoor_score, label

def normalize_category(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    if "전시" in s or "exhibition" in s: return "전시"
    if "공연" in s or "performance" in s or "콘서트" in s or "연극" in s: return "공연"
    if "축제" in s or "festival" in s or "페스티벌" in s: return "축제"
    return s

# -----------------------
# 평판/인기도
# -----------------------
def popularity_score(p: Dict[str, Any]) -> float:
    n = norm_post(p)
    vc = float(n["view_count"] or 0)
    lc = float(n["like_count"] or 0)
    return (math.log1p(vc)*0.3 + math.log1p(lc)*0.7) / 10.0

def rating_tuple(p: Dict[str, Any]):
    n = norm_post(p)
    avg_rating = to_float(n["avg_rating"], None)
    review_count = int(n["review_count"] or 0)
    if avg_rating is None:
        acc = 0.0; cnt = 0
        for rv in (n["reviews"] or []):
            r = to_float(rv.get("rating"))
            if r is not None: acc += r; cnt += 1
        if cnt > 0:
            avg_rating = acc / cnt
            review_count = max(review_count, cnt)
        else:
            avg_rating = 0.0
    return (avg_rating or 0.0), review_count

# -----------------------
# KMA (PTY: 강수형태)
# -----------------------
def parse_pty(pty: Any) -> Optional[int]:
    if pty is None: return None
    try:
        return int(str(pty).strip())
    except Exception:
        return None

def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    RE = 6371.00877; GRID = 5.0
    SLAT1 = 30.0; SLAT2 = 60.0; OLON = 126.0; OLAT = 38.0
    XO = 210/GRID; YO = 675/GRID
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD; slat2 = SLAT2 * DEGRAD
    olon  = OLON  * DEGRAD; olat  = OLAT  * DEGRAD
    sn = math.log(math.cos(slat1)/math.cos(slat2)) / math.log(math.tan(math.pi*0.25 + slat2*0.5)/math.tan(math.pi*0.25 + slat1*0.5))
    sf = (math.tan(math.pi*0.25 + slat1*0.5) ** sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi*0.25 + olat*0.5) ** sn)
    ra = re * sf / (math.tan(math.pi*0.25 + lat*DEGRAD*0.5) ** sn)
    theta = lon*DEGRAD - OLON*DEGRAD
    if theta > math.pi:  theta -= 2.0*math.pi
    if theta < -math.pi: theta += 2.0*math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO
    y = ro - ra * math.cos(theta) + YO
    return int(x + 1.5), int(y + 1.5)

def _vilage_base(now: datetime | None = None) -> Tuple[str, str]:
    now = now or _now_kr()
    sched = [2,5,8,11,14,17,20,23]
    ref = now - timedelta(minutes=10)
    ymd = ref.strftime("%Y%m%d"); h = ref.hour
    bt = None
    for hh in reversed(sched):
        if h >= hh:
            bt = hh; break
    if bt is None:
        prev = ref - timedelta(days=1)
        ymd = prev.strftime("%Y%m%d"); bt = 23
    return ymd, f"{bt:02d}00"

def fetch_pty_from_kma(lat: float, lon: float, for_date: date) -> Optional[int]:
    if not WEATHER_KEY: return None
    nx, ny = latlon_to_grid(float(lat), float(lon))
    base_date, base_time = _vilage_base()
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": WEATHER_KEY,
        "numOfRows": 300, "pageNo": 1, "dataType": "JSON",
        "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        j = r.json()
        items = (((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
        ptys = [it for it in items if str(it.get("category")) == "PTY"]
        if not ptys: return None
        tgt = for_date.strftime("%Y%m%d")
        target_hour = _now_kr().hour if for_date == _today_kr() else 12
        target_str = f"{target_hour:02d}00"
        cand = [it for it in ptys if str(it.get("fcstDate")) == tgt] or ptys
        def _dist(t: str) -> int:
            try: return abs(int(t) - int(target_str))
            except Exception: return 9999
        chosen = min(cand, key=lambda it: _dist(str(it.get("fcstTime","0000"))))
        return int(str(chosen.get("fcstValue")).strip())
    except Exception:
        return None

def weather_suitability(pty: Optional[int], indoor_score: float, outdoor_score: float) -> float:
    if pty is None: return 0.0
    if pty == 0:    return max(indoor_score, outdoor_score)
    return indoor_score

# -----------------------
# 가중치/윈도우/시간점수
# -----------------------
def spotlight_weights(mode: str, weather_active: bool) -> Dict[str, float]:
    if mode == "DAILY":
        return dict(time=0.25, weather=(0.15 if weather_active else 0.0),
                    distance=0.20, interest=0.15, popularity=0.15, rating=0.10)
    return dict(time=0.20, weather=(0.05 if weather_active else 0.0),
                distance=0.15, interest=0.20, popularity=0.25, rating=0.15)

def in_window(p: Dict[str, Any], mode: str, today: date, month_first: date, month_last: date) -> bool:
    n = norm_post(p)
    sd = parse_date(n["start_date"]); ed = parse_date(n["end_date"])
    if sd is None and ed is None:
        return True
    if mode == "DAILY":
        d = today
        if sd and d < sd: return False
        if ed and d > ed: return False
        return True
    s = sd or month_first
    e = ed or month_last
    return not (e < month_first or s > month_last)

def time_score(p: Dict[str, Any], mode: str, today: date) -> float:
    n = norm_post(p)
    sd = parse_date(n["start_date"]); ed = parse_date(n["end_date"])
    if mode == "DAILY":
        if sd and ed and sd <= today <= ed: return 1.0
        if (sd and sd == today) or (ed and ed == today): return 0.9
        return 0.5
    if not sd: return 0.5
    delta = abs((sd - today).days)
    if delta <= 3: return 1.0
    if delta <= 7: return 0.85
    if delta <= 14: return 0.7
    return 0.55

# -----------------------
# 개별 게시글 점수 계산
# -----------------------
def score_post_spotlight(
    p: Dict[str, Any], mode: str, center: Optional[Dict[str, float]],
    radius_km: Optional[float], interests: List[str],
    pty: Optional[int], weights: Dict[str, float], today: date
) -> Optional[Tuple[float, Any]]:
    mfirst, mlast = _month_window_kr()
    if not in_window(p, mode, today, mfirst, mlast):
        return None

    n = norm_post(p)
    if n["id"] is None:    # ❗ id 없으면 채택하지 않음
        return None

    t_score = time_score(p, mode, today)

    # 거리(좌표 없으면 0)
    dist_score = 0.0
    if center and n["gps_y"] is not None and n["gps_x"] is not None:
        lat = to_float(n["gps_y"]); lon = to_float(n["gps_x"])
        if lat is not None and lon is not None:
            dist_km = haversine_km(center["lat"], center["lon"], lat, lon)
            rk = float(radius_km or 10)
            if dist_km <= rk:
                dist_score = max(0.0, 1 - dist_km/max(rk,1e-6))

    # 관심 카테고리
    cat_post = normalize_category(n["category"])
    interests_norm = {normalize_category(c) for c in (interests or []) if c}
    interest_score = 1.0 if (cat_post and cat_post in interests_norm) else 0.0

    blob = collect_text(p)
    indoor_s, outdoor_s, _ = detect_inout(blob)
    weather_sc = weather_suitability(pty, indoor_s, outdoor_s)

    pop_sc = popularity_score(p)
    avg_rating, review_count = rating_tuple(p)
    rating_sc = (avg_rating/5.0) * (1 - math.exp(-review_count/5.0)) if avg_rating else 0.0

    score = (
        t_score       * weights["time"]      +
        weather_sc    * weights["weather"]   +
        dist_score    * weights["distance"]  +
        interest_score* weights["interest"]  +
        pop_sc        * weights["popularity"]+
        rating_sc     * weights["rating"]
    )

    return (round(score,4), n["id"])

# -----------------------
# 메인: 상단 N개 ID만 반환 (정수)
# -----------------------
def pick_ids_json(
    mode: str, posts: List[Dict[str, Any]], top_k: int = 10,
    interest_categories: List[str] | None = None,
    center: Dict[str, float] | None = None,
    radius_km: float | None = None,
    weather: Any | None = None,
    auto_weather: bool = True
) -> Dict[str, List[int]]:
    mode = (mode or "DAILY").upper()
    if mode not in ("DAILY","MONTHLY"):
        mode = "DAILY"

    today = _today_kr()

    pty = parse_pty(weather)
    if pty is None and auto_weather and center and "lat" in center and "lon" in center:
        try:
            pty = fetch_pty_from_kma(center["lat"], center["lon"], today)
        except Exception:
            pty = None

    weights = spotlight_weights(mode, weather_active=(pty is not None))
    interests = interest_categories or []
    rk = radius_km or 10.0

    scored: List[Tuple[float, Any]] = []
    for p in posts:
        s = score_post_spotlight(p, mode, center, rk, interests, pty, weights, today)
        if s:
            score, pid = s
            if pid is not None:
                scored.append((score, pid))

    scored.sort(key=lambda x: x[0], reverse=True)

    ids: List[int] = []
    for _, pid in scored[:max(1, int(top_k))]:
        try:
            ids.append(int(pid))
        except Exception:
            continue

    return {"recommendedCultureIds": ids}
