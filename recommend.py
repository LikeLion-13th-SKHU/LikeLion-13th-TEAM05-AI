# recommend.py — posts 스키마 자동매핑 + 자연어 요약 그대로 유지
import math, re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import re, html

def parse_date(s: Optional[str]) -> Optional[date]:
    if not s: return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try: return datetime.strptime(s[:len(fmt)], fmt).date()
        except Exception: pass
    return None

def to_float(x: Any, default: Optional[float]=None) -> Optional[float]:
    try: return float(x)
    except Exception: return default

def tokenize(text: str) -> List[str]:
    if not text: return []
    return re.findall(r"[가-힣a-z0-9]+", text.lower())

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# 🔁 스키마 정규화
def norm_post(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":            p.get("id") or p.get("postId"),
        "title":         p.get("title"),
        "content":       p.get("contents") or p.get("content") or "",
        "area":          p.get("area") or "",
        "sigungu":       p.get("sigungu") or "",
        "category":      p.get("category") or p.get("category_name") or "",
        "start_date":    p.get("start_date") or p.get("startDate"),
        "end_date":      p.get("end_date") or p.get("endDate"),
        "view_count":    p.get("view_count") or p.get("viewCount") or 0,
        "like_count":    p.get("like_count") or p.get("likeCount") or 0,
        "place":         p.get("place") or "",
        "placeAddr":     p.get("placeAddr") or p.get("address") or "",
        "gps_x":         p.get("gps_x") or p.get("lng"),
        "gps_y":         p.get("gps_y") or p.get("lat"),
        "avg_rating":    p.get("avg_rating"),
        "review_count":  p.get("review_count") or 0,
        "reviews":       p.get("reviews") or [],
    }

EMO_MAP = {
    "조용한": ["조용","차분","한적","고요","잔잔","힐링","명상","서정","잔잔한"],
    "신나는": ["신나","라이브","댄스","밴드","페스티벌","흥겨운","업템포","디제잉","EDM","불꽃"],
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

def parse_pty(pty: Any) -> Optional[int]:
    if pty is None: return None
    try: return int(str(pty).strip())
    except Exception: return None

def weather_suitability(pty: Optional[int], indoor_score: float, outdoor_score: float) -> float:
    if pty is None: return 0.0
    if pty == 0:    return max(indoor_score, outdoor_score)
    return indoor_score

def emotion_score(target_emotion: Optional[str], blob: str) -> float:
    if not target_emotion: return 0.0
    keys = EMO_MAP.get(target_emotion, [])
    if not keys: return 0.0
    hits = sum(1 for k in keys if k in blob)
    if hits <= 0: return 0.0
    if hits == 1: return 0.5
    if hits == 2: return 0.7
    return 1.0

def normalize_category(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    if "전시" in s or "exhibition" in s: return "전시"
    if "공연" in s or "performance" in s or "콘서트" in s or "연극" in s: return "공연"
    if "축제" in s or "festival" in s or "페스티벌" in s: return "축제"
    return s

def get_weights(weather_active: bool) -> Dict[str, float]:
    if weather_active:
        return dict(category=0.20, distance=0.20, text=0.18, popularity=0.12, rating=0.06, emotion=0.08, weather=0.16)
    else:
        return dict(category=0.24, distance=0.24, text=0.22, popularity=0.14, rating=0.08, emotion=0.08, weather=0.00)

def build_query_terms(message: str, keywords: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    interests = []
    if user and user.get("interest_tags"):
        interests = [t.strip().lower() for t in re.split(r"[,\s/]+", user["interest_tags"]) if t.strip()]
    return {
        "when": keywords.get("when"),
        "where": keywords.get("where"),
        "category": normalize_category(keywords.get("Category") or keywords.get("category")),
        "emotion": keywords.get("emotion"),
        "radius_km": float(keywords.get("radius_km") or 5),
        "weather": keywords.get("weather"),
        "terms": list(set(tokenize(message) + interests))
    }

def compute_review_stats(p: Dict[str, Any]):
    n = norm_post(p)
    avg_rating = to_float(n["avg_rating"], None)
    review_count = int(n["review_count"] or 0)
    if avg_rating is None:
        acc = 0.0; nrv = 0
        for rv in (n["reviews"] or []):
            r = to_float(rv.get("rating"))
            if r is not None: acc += r; nrv += 1
        if nrv > 0:
            avg_rating = acc / nrv
            review_count = max(review_count, nrv)
        else:
            avg_rating = 0.0
    return avg_rating or 0.0, review_count

def score_post(p: Dict[str, Any], center: Optional[Dict[str, float]], q: Dict[str, Any], weights: Dict[str, float]):
    n = norm_post(p)

    # 날짜 필터
    when = q.get("when")
    if when:
        day = parse_date(when) or date.today()
        sd = parse_date(n["start_date"]); ed = parse_date(n["end_date"])
        if sd and day < sd: return None
        if ed and day > ed: return None

    # 카테고리
    cat_target = q.get("category")
    cat_post = normalize_category(n["category"])
    cat_score = 1.0 if (cat_target and cat_post and cat_target in str(cat_post)) else 0.0

    # 거리: 좌표 있으면 좌표, 없으면 지역/시군구 텍스트 매칭
    dist_score = 0.0; dist_km = None
    if center and n["gps_y"] is not None and n["gps_x"] is not None:
        lat = to_float(n["gps_y"]); lon = to_float(n["gps_x"])
        if lat is not None and lon is not None:
            dist_km = haversine_km(center["lat"], center["lon"], lat, lon)
            rk = float(q.get("radius_km") or 5)
            if dist_km <= rk: dist_score = max(0.0, 1 - dist_km/max(rk,1e-6))
    else:
        where = (q.get("where") or "").lower()
        if where:
            txt = " ".join([str(n["area"] or ""), str(n["sigungu"] or ""), str(n["placeAddr"] or ""), str(n["place"] or "")]).lower()
            if any(w in txt for w in where.split() if w): dist_score = 0.6

    # 텍스트/감성/날씨
    terms = set(q.get("terms") or [])
    blob = collect_text(p)
    hits = sum(1 for t in terms if t and t in blob)
    text_score = min(1.0, hits / max(3, len(terms) or 1))

    indoor_s, outdoor_s, inout_label = detect_inout(blob)
    pty = parse_pty(q.get("weather"))
    weather_score = weather_suitability(pty, indoor_s, outdoor_s)

    emo_score = 0.0  # (옵션) emotion을 이 모듈에서 사용하려면 FE에 전달
    # emo_score = emotion_score(q.get("emotion"), blob)

    # 인기도/평점
    vc = float(n["view_count"] or 0); lc = float(n["like_count"] or 0)
    pop_score = (math.log1p(vc)*0.3 + math.log1p(lc)*0.7) / 10.0
    avg_rating, review_count = compute_review_stats(p)
    rating_score = (avg_rating/5.0) * (1 - math.exp(-review_count/5.0)) if avg_rating else 0.0

    score = (
        cat_score      * weights["category"]  +
        dist_score     * weights["distance"]  +
        text_score     * weights["text"]      +
        pop_score      * weights["popularity"]+
        rating_score   * weights["rating"]    +
        emo_score      * weights["emotion"]   +
        weather_score  * weights["weather"]
    )

    item = {
        "id": n["id"],
        "title": n["title"],
        "addr": n["placeAddr"],
        "place": n["place"],
        "area": n["area"],
        "sigungu": n["sigungu"],
        "category": cat_post,
        "distance_km": round(dist_km,2) if dist_km is not None else None,
        "avg_rating": round(avg_rating,2) if avg_rating else 0.0,
        "review_count": review_count,
        "view_count": int(vc),
        "like_count": int(lc),
        "inout": inout_label,
        "pty": pty,
        "score": round(score,4),
    }
    return score, item

def get_weights(weather_active: bool) -> Dict[str, float]:
    if weather_active:
        return dict(category=0.20, distance=0.20, text=0.18, popularity=0.12, rating=0.06, emotion=0.08, weather=0.16)
    else:
        return dict(category=0.24, distance=0.24, text=0.22, popularity=0.14, rating=0.08, emotion=0.08, weather=0.00)

def build_query_terms(message: str, keywords: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    interests = []
    if user and user.get("interest_tags"):
        interests = [t.strip().lower() for t in re.split(r"[,\s/]+", user["interest_tags"]) if t.strip()]
    return {
        "when": keywords.get("when"),
        "where": keywords.get("where"),
        "category": normalize_category(keywords.get("Category") or keywords.get("category")),
        "emotion": keywords.get("emotion"),
        "radius_km": float(keywords.get("radius_km") or 5),
        "weather": keywords.get("weather"),
        "terms": list(set(tokenize(message) + interests))
    }

def recommend_json(message: str, keywords: Dict[str, Any], posts: List[Dict[str, Any]],
                   user: Dict[str, Any] | None = None, center: Dict[str, float] | None = None,
                   radius_km: float | None = None, top_k: int = 5) -> Dict[str, Any]:
    q = build_query_terms(message, keywords or {}, user or {})
    if radius_km is not None: q["radius_km"] = float(radius_km)
    weather_active = (q.get("weather") is not None)
    weights = get_weights(weather_active)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for p in posts:
        s = score_post(p, center, q, weights)
        if s: scored.append(s)
    scored.sort(key=lambda x: x[0], reverse=True)
    items = [x[1] for x in scored[:max(1, top_k)]]
    return {"items": items, "debug": {"query": q, "weights": weights, "count_in": len(posts), "count_scored": len(scored)}}

def render_titles_plain(payload: Dict[str, Any]) -> str:
    """
    요청: {"message":"...", "result":{"items":[{"id":..,"title":"..."}, ...]}}
    응답: "제목1, 제목2, 제목3"   # 줄바꿈 없음, 쉼표만 구분자, 특수문자 제거(, 제외)
    """
    result = payload.get("result") or {}
    items  = result.get("items") or []
    if not items:
        return "추천 가능한 게시글이 없어요."

    titles: List[str] = []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t:
            fid = it.get("id")
            t = f"ID{fid}" if fid is not None else ""
        if t:
            titles.append(t)

    # 1) 기본 조인
    s = ", ".join(titles)

    # 2) HTML 엔티티 해제 (예: &#39; -> ')
    s = html.unescape(s)

    # 3) 특수문자 제거: 한글/영문/숫자/공백/쉼표만 허용
    #    대괄호, 따옴표 등은 제거됨
    s = re.sub(r"[^가-힣a-zA-Z0-9 ,]", "", s)

    # 4) 쉼표 주변 공백/중복 정리
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(parts)
