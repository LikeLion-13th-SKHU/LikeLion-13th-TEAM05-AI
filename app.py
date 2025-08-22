# app.py — keep /ask AND add /day_month_pik
import os
from typing import Any, Dict, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

def verify(x_internal_token: str | None = Header(None)):
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

# === Imports ===
try:
    from discriminate import classify_intent
except Exception:
    classify_intent = None

try:
    from recommend import recommend_json, render_answer_json
except Exception:
    recommend_json = None
    render_answer_json = None

try:
    from day_month_pik import pick_ids_json
except Exception:
    pick_ids_json = None

app = FastAPI(title="ArtPick AI (/ask + /day_month_pik)", version="6.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===== Schemas =====
class Center(BaseModel):
    lat: float
    lon: float

# --- /ask ---
class AskIn(BaseModel):
    message: str
    posts: List[Dict[str, Any]] = Field(..., description="추천 후보 게시글 배열(필수)")
    meta: Dict[str, Any] | None = None
    user: Dict[str, Any] | None = None
    center: Center | None = None
    radius_km: float | None = None
    top_k: int = 5

class AnswerOut(BaseModel):
    answer: str

# --- /day_month_pik ---
class DayMonthPikIn(BaseModel):
    mode: Literal["DAILY", "MONTHLY"] = Field(..., description="DAILY 또는 MONTHLY") 
    posts: List[Dict[str, Any]] = Field(..., description="후보 게시글 배열(필수)")
    top_k: int = 10
    interest_categories: List[str] | None = Field(default=None, description="예: ['전시','공연']")
    center: Center | None = Field(default=None, description="거리/날씨용 중심 좌표")
    radius_km: float | None = Field(default=None, description="거리 반경(기본 10km)")
    weather: str | int | None = Field(default=None, description="PTY 코드(옵션)")
    auto_weather: bool = Field(default=True, description="weather 미제공 + center 있을 때 자동 조회")

class DayMonthPikOut(BaseModel):
    recommendedCultureIds: List[Any]

# ===== Routes =====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerOut, dependencies=[Depends(verify)])
def ask(body: AskIn):
    """
    항상 posts를 받아 추천 실행.
    - 의도 분류(classify_intent)로 where/when/emotion/weather 추출(날씨 자동 보강은 discriminate.py에 구현됨)
    - 결과는 항상 {"answer": "..."} 한 가지만 반환
    """
    if not body.posts:
        return AnswerOut(answer="게시글 목록(posts)이 비어 있어요. 후보 목록을 함께 보내주세요.")
    if classify_intent is None:
        raise HTTPException(500, "discriminate.classify_intent가 없습니다.")
    if recommend_json is None:
        raise HTTPException(500, "recommend.recommend_json가 없습니다.")

    # 1) 의도/키워드 추출
    data = classify_intent({"message": body.message, "meta": body.meta or {}}) or {}
    kws = data.get("keywords") or {}

    # 2) 기본값 보정
    if "radius_km" not in kws or kws.get("radius_km") is None:
        kws["radius_km"] = body.radius_km if body.radius_km is not None else 5
    if not kws.get("when"):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        kws["when"] = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")

    # 3) 중심 좌표: body.center > meta.center > 없음
    center_dict: Optional[Dict[str, float]] = body.center.dict() if body.center else None
    if not center_dict and body.meta and isinstance(body.meta.get("center"), dict):
        center_dict = body.meta["center"]

    # 4) 추천 실행
    result = recommend_json(
        message=body.message,
        keywords=kws,
        posts=body.posts,
        user=body.user or ((body.meta or {}).get("user")),
        center=center_dict,
        radius_km=kws.get("radius_km"),
        top_k=body.top_k,
    )

    # 5) 자연어 요약만 반환
    if render_answer_json:
        wrapped = render_answer_json({"message": body.message, "result": result}) or {}
        answer = wrapped.get("answer") or "추천 결과를 확인하세요."
    else:
        answer = "추천 결과를 확인하세요."
    return AnswerOut(answer=str(answer))

@app.post("/day_month_pik", response_model=DayMonthPikOut, dependencies=[Depends(verify)])
def day_month_pik_route(body: DayMonthPikIn):
    if pick_ids_json is None:
        raise HTTPException(500, "day_month_pik.pick_ids_json가 없습니다.")
    if not body.posts:
        return DayMonthPikOut(recommendedCultureIds=[])

    result = pick_ids_json(
        mode=body.mode,
        posts=body.posts,
        top_k=body.top_k,
        interest_categories=body.interest_categories or [],
        center=(body.center.dict() if body.center else None),
        radius_km=body.radius_km,
        weather=body.weather,
        auto_weather=body.auto_weather,
    )
    # 이미 {"recommendedCultureIds":[...]} 형태
    return DayMonthPikOut(**result)
