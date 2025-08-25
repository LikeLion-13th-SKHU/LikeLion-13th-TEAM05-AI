import re, html, math
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

def _now_kr() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _tokens(s: str) -> List[str]:
    if not s: return []
    return re.findall(r"[가-힣a-z0-9]+", s.lower())

def _normalize_category(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower()
    if "전시" in s or "exhibition" in s: return "전시"
    if "공연" in s or "performance" in s or "콘서트" in s or "뮤지컬" in s or "연극" in s: return "공연"
    if "축제" in s or "festival" in s or "페스티벌" in s: return "축제"
    return s

def _date_to_yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def _parse_date(s: Any) -> Optional[datetime]:
    if not s: return None
    v = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(v[:len(fmt)], fmt).replace(tzinfo=ZoneInfo("Asia/Seoul"))
        except Exception:
            pass
    return None

def _concat_post_text(p: Dict[str, Any]) -> str:
    parts = [
        str(p.get("title") or ""),
        str(p.get("contents") or p.get("content") or ""),
        str(p.get("area") or ""),
        str(p.get("sigungu") or ""),
        str(p.get("category") or ""),
    ]
    for rv in (p.get("reviews") or []):
        parts.append(str(rv.get("content") or rv.get("contents") or ""))
    return _clean(" ".join(parts)).lower()

def _similarity(q: str, text: str) -> float:
    a = set(_tokens(q)); b = set(_tokens(text))
    if not a or not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

def _popularity(p: Dict[str, Any]) -> float:
    vc = float(p.get("viewCount") or p.get("view_count") or 0)
    lc = float(p.get("likeCount") or p.get("like_count") or 0)
    return (math.log1p(vc)*0.3 + math.log1p(lc)*0.7) / 10.0

def _rating_tuple(p: Dict[str, Any]) -> Tuple[float, int]:
    avg = p.get("avg_rating")
    cnt = int(p.get("review_count") or 0)
    if avg is None:
        acc = 0.0; c = 0
        for rv in (p.get("reviews") or []):
            r = rv.get("rating")
            try:
                r = float(r)
            except Exception:
                r = None
            if r is not None:
                acc += r; c += 1
        if c > 0:
            avg = acc / c
            cnt = max(cnt, c)
        else:
            avg = 0.0
    return float(avg or 0.0), cnt

def _time_score(p: Dict[str, Any], today: datetime) -> float:
    sd = _parse_date(p.get("startDate") or p.get("start_date"))
    ed = _parse_date(p.get("endDate") or p.get("end_date"))
    t = today.date()
    if sd and ed and sd.date() <= t <= ed.date(): return 1.0
    if (sd and sd.date() == t) or (ed and ed.date() == t): return 0.9
    return 0.5

def _score_post(message: str, kws: Dict[str, Any], user: Dict[str, Any], p: Dict[str, Any]) -> float:
    today = _now_kr()
    text = _concat_post_text(p)
    sim = _similarity(message, text)
    pop = _popularity(p)
    avg, rcnt = _rating_tuple(p)
    rating = (avg/5.0) * (1 - math.exp(-rcnt/5.0)) if avg else 0.0
    cat_q = _normalize_category(kws.get("Category"))
    cat_p = _normalize_category(p.get("category"))
    cmatch = 1.0 if (cat_q and cat_p and cat_q == cat_p) else 0.0
    its = set((user.get("interest_tags") or "").split(",")) if user.get("interest_tags") else set()
    imatch = 1.0 if (cat_p and cat_p in its) else 0.0
    tscore = _time_score(p, today)
    return sim*0.35 + cmatch*0.2 + imatch*0.15 + pop*0.15 + rating*0.15 + tscore*0.1

def recommend_json(message: str, keywords: Dict[str, Any], posts: List[Dict[str, Any]], user: Dict[str, Any] | None = None, top_k: int = 5) -> Dict[str, Any]:
    user = user or {}
    scores: List[Tuple[float, Dict[str, Any]]] = []
    for p in posts:
        t = (p.get("title") or "").strip()
        if not t: 
            continue
        s = _score_post(message, keywords or {}, user, p)
        scores.append((s, p))
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in scores[:max(1, int(top_k))]]
    items: List[Dict[str, Any]] = []
    for p in selected:
        items.append({
            "id": p.get("id") or p.get("postId"),
            "title": p.get("title"),
            "area": p.get("area"),
            "sigungu": p.get("sigungu"),
            "category": p.get("category"),
            "startDate": p.get("startDate") or p.get("start_date"),
            "endDate": p.get("endDate") or p.get("end_date"),
        })
    return {"items": items}

def _clean_title(t: str) -> str:
    t = html.unescape((t or "").strip())
    t = re.sub(r"[^가-힣a-zA-Z0-9 ,]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _has_jongsung(ch: str) -> bool:
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False

def _choose_eul_reul(text: str) -> str:
    for ch in reversed(text):
        if '가' <= ch <= '힣':
            return "을" if _has_jongsung(ch) else "를"
    return "를"

def render_titles_sentence(payload: Dict[str, Any]) -> str:
    result = payload.get("result") or {}
    items  = result.get("items") or []
    if not items:
        return "추천 가능한 게시글이 없어요."
    titles: List[str] = []
    for it in items:
        t = _clean_title(it.get("title") or "")
        if not t:
            fid = it.get("id")
            t = f"ID{fid}" if fid is not None else ""
        if t:
            titles.append(t)
    joined = ", ".join(p for p in (s.strip() for s in titles) if p)
    particle = _choose_eul_reul(titles[-1]) if titles else "를"
    return f"{joined}{particle} 추천드립니다."

