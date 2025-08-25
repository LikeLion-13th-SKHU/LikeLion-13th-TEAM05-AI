# discriminate.py
import os, re, json, math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import requests

load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ.get("BASE_URL","https://api.together.xyz"))
MODEL_ID = os.environ.get("MODEL_ID","meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

WEATHER_KEY = os.environ.get("weather_API_KEY")         # 기상청 단기예보 인증키
KAKAO_KEY   = os.environ.get("KAKAO_REST_API_KEY")      # (선택) 카카오 로컬 API 키

JSON_BLOCK = re.compile(r"\{[\s\S]*\}")

SYSTEM = (
    "너는 의도 분류기야. 입력(JSON)만 보고 다음 JSON 스키마로만 출력해.\n"
    "{\n"
    '  "function": 1|2|3,                        // 1=기능 설명, 2=추천, 3=반려\n'
    '  "keywords": {                             // function=2일 때만 유효\n'
    '     "when": string|null,\n'
    '     "where": string|null,\n'
    '     "Category": "축제"|"전시"|"공연"|null,\n'
    '     "weather": string|null,\n'
    '     "emotion": string|null,\n'
    '     "radius_km": number|null\n'
    '  }|null,\n'
    '  "message": string|null,                   // function=3일 때 반려 사유\n'
    '  "original_message": string|null           // function=2일 때 사용자 질문 원문(에코)\n'
    "}\n"
    "- 임의로 지어내지 말 것. 모르면 null.\n"
    "- 오직 JSON만 출력(코드블록 금지)."
)

FEWSHOTS: List[Dict[str, str]] = [
    {"role":"user","content":json.dumps({"message":"아트픽에서 위치 기반 검색은 어떻게 해?"}, ensure_ascii=False)},
    {"role":"assistant","content":json.dumps({"function":1,"keywords":None,"message":None,"original_message":None}, ensure_ascii=False)},
    {"role":"user","content":json.dumps({"message":"서울 강남구 이번 주말 전시 추천해줘"} , ensure_ascii=False)},
    {"role":"assistant","content":json.dumps({"function":2,"keywords":{"when":None,"where":"서울 강남구","Category":"전시","weather":None,"emotion":None,"radius_km":5},"message":None,"original_message":"서울 강남구 이번 주말 전시 추천해줘"}, ensure_ascii=False)},
    {"role":"user","content":json.dumps({"message":"나 잘생겼지?"}, ensure_ascii=False)},
    {"role":"assistant","content":json.dumps({"function":3,"keywords":None,"message":"서비스와 무관한 질문입니다.","original_message":None}, ensure_ascii=False)},
]

# -----------------------------
# 공통 유틸
# -----------------------------
def _safe_json(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        m = JSON_BLOCK.search(txt or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def _now_kr() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def _today_kr() -> str:
    return _now_kr().strftime("%Y%m%d")

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s: return None
    s = str(s).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try: return datetime.strptime(s[:len(fmt)], fmt).date()
        except Exception: pass
    return None

# -----------------------------
# 지오코딩 (Kakao → 실패 시 Nominatim)
# -----------------------------
def _geocode_kakao(query: str) -> Optional[Tuple[float, float]]:
    if not KAKAO_KEY: return None
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    try:
        # 1) 장소/키워드
        r = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            params={"query": query, "size": 1},
            headers=headers, timeout=5
        )
        if r.ok and r.json().get("documents"):
            doc = r.json()["documents"][0]
            return float(doc["y"]), float(doc["x"])  # (lat, lon)

        # 2) 주소
        r = requests.get(
            "https://dapi.kakao.com/v2/local/search/address.json",
            params={"query": query},
            headers=headers, timeout=5
        )
        if r.ok and r.json().get("documents"):
            doc = r.json()["documents"][0]
            return float(doc["y"]), float(doc["x"])
    except Exception:
        pass
    return None

def _geocode_nominatim(query: str) -> Optional[Tuple[float, float]]:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": "ArtPick/1.0"},
            timeout=8
        )
        arr = r.json() if r.ok else []
        if arr:
            return float(arr[0]["lat"]), float(arr[0]["lon"])
    except Exception:
        pass
    return None

def geocode_any(query: str) -> Optional[Tuple[float, float]]:
    return _geocode_kakao(query) or _geocode_nominatim(query)

# -----------------------------
# KMA 좌표 변환 (위경도 -> 격자 nx,ny)
# -----------------------------
def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    RE = 6371.00877  # 지구반경(km)
    GRID = 5.0       # 격자간격(km)
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 210/GRID
    YO = 675/GRID

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD; slat2 = SLAT2 * DEGRAD
    olon  = OLON  * DEGRAD; olat  = OLAT  * DEGRAD

    sn = math.log(math.cos(slat1)/math.cos(slat2)) / math.log(math.tan(math.pi*0.25 + slat2*0.5)/math.tan(math.pi*0.25 + slat1*0.5))
    sf = (math.tan(math.pi*0.25 + slat1*0.5) ** sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi*0.25 + olat*0.5) ** sn)

    ra = re * sf / (math.tan(math.pi*0.25 + lat*DEGRAD*0.5) ** sn)
    theta = (lon*DEGRAD - olon)
    if theta > math.pi:  theta -= 2.0*math.pi
    if theta < -math.pi: theta += 2.0*math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO
    y = ro - ra * math.cos(theta) + YO
    return int(x + 1.5), int(y + 1.5)

# -----------------------------
# 단기예보 base_date/base_time 선택
#  - 발표시각: 02,05,08,11,14,17,20,23
#  - 제공은 base_time + 10분 이후
# -----------------------------
def _vilage_base(now: datetime | None = None) -> Tuple[str, str]:
    now = now or _now_kr()
    sched = [2,5,8,11,14,17,20,23]
    ref = now - timedelta(minutes=10)  # 제공 오차 고려
    ymd = ref.strftime("%Y%m%d")
    h = ref.hour
    bt = None
    for hh in reversed(sched):
        if h >= hh:
            bt = hh; break
    if bt is None:
        prev = ref - timedelta(days=1)
        ymd = prev.strftime("%Y%m%d")
        bt = 23
    return ymd, f"{bt:02d}00"

# -----------------------------
# 단기예보 PTY 조회 (getVilageFcst)
# -----------------------------
def _kma_get_pty(nx: int, ny: int, when: str, target_hour: Optional[int] = None) -> Optional[int]:
    if not WEATHER_KEY:
        return None
    base_date, base_time = _vilage_base()
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": WEATHER_KEY,
        "numOfRows": 300,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        j = r.json()
        items = (((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
        ptys = [it for it in items if str(it.get("category")) == "PTY"]
        if not ptys:
            return None

        # when(YYYYMMDD)에 맞는 fcst 중 target_hour 근접 선택
        if target_hour is None:
            target_hour = _now_kr().hour if _today_kr() == when else 12
        target_str = f"{target_hour:02d}00"
        cand = [it for it in ptys if str(it.get("fcstDate")) == when] or ptys

        def _dist(t: str) -> int:
            try: return abs(int(t) - int(target_str))
            except Exception: return 9999

        chosen = min(cand, key=lambda it: _dist(str(it.get("fcstTime","0000"))))
        return int(str(chosen.get("fcstValue")).strip())
    except Exception:
        return None

# -----------------------------
# 의도 분류 + 자동 날씨/위치 보강
# -----------------------------
def classify_intent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: {"message": "...", "meta": {...}}  // JSON in
    return:  {"function":..., "keywords":..., "message":..., "original_message": ...}  // JSON out

    - function=2(추천) && keywords.when이 비면 한국시간 오늘(YYYYMMDD)
    - function=2 && keywords.weather가 비어있으면:
        1) meta.center(lat,lon) 또는 meta.nx,ny 사용
        2) 없으면 keywords.where(장소 문자열) → 지오코딩으로 lat,lon 추정
        3) lat,lon → nx,ny 변환 후 KMA PTY 호출 → keywords.weather 채움
    """
    messages = [{"role":"system","content":SYSTEM}] + FEWSHOTS + [
        {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
    ]
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.0,
        response_format={"type":"json_object"}
    )

    data = _safe_json(resp.choices[0].message.content)

    # 기본 보정
    data.setdefault("function", 3)
    data.setdefault("keywords", None)
    data.setdefault("message", None)
    data.setdefault("original_message", None)

    if int(data.get("function", 3)) == 2:
        kws = dict(data.get("keywords") or {})

        # when 보정
        when = (kws.get("when") if isinstance(kws, dict) else None) or data.get("when")
        if when is None or str(when).strip() in ("", "null"):
            when = _today_kr()
        else:
            d = _parse_date(when)
            if d: when = d.strftime("%Y%m%d")
        kws["when"] = when

        # 반경 기본값
        if kws.get("radius_km") is None:
            kws["radius_km"] = 5

        # ----- weather(PTY) 자동 채움 -----
        if kws.get("weather") in (None, "", "null"):
            meta = payload.get("meta") or {}
            nx, ny = meta.get("nx"), meta.get("ny")
            lat = lon = None

            # 1) center 우선
            if nx is None or ny is None:
                c = meta.get("center") or {}
                lat = c.get("lat"); lon = c.get("lon")

            # 2) center 없으면 where 지오코딩
            if (lat is None or lon is None) and (nx is None or ny is None):
                where_txt = kws.get("where") or meta.get("where") or (payload.get("message") if isinstance(payload.get("message"), str) else None)
                if where_txt:
                    try:
                        geo = geocode_any(where_txt)
                        if geo: lat, lon = geo
                    except Exception:
                        pass

            # 3) 격자 계산
            if (nx is None or ny is None) and (lat is not None and lon is not None):
                try:
                    nx, ny = latlon_to_grid(float(lat), float(lon))
                except Exception:
                    nx = ny = None

            # 4) KMA PTY 조회
            if nx is not None and ny is not None:
                pty = _kma_get_pty(int(nx), int(ny), when)
                if pty is not None:
                    kws["weather"] = str(pty)

        data["keywords"] = kws

        # 사용자 질문 원문 에코
        try:
            data["original_message"] = (payload or {}).get("message")
        except Exception:
            data["original_message"] = None

    return data
