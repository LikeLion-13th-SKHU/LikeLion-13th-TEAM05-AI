# app.py — trimmed
import os, hashlib, pathlib
from typing import Any, Dict, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

def verify(x_internal_token: str | None = Header(None)):
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

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

app = FastAPI(title="ArtPick AI (new field names)", version="6.5.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

APP_SIG = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[ArtPick] app.py md5 = {APP_SIG}")

class AskIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    message: str
    cultures: List[Dict[str, Any]] = Field(default_factory=list)
    requestCount: int = 5
    userInterests: List[str] | None = None
    user: Dict[str, Any] | None = None
    userId: int | None = None
    timestamp: str | None = None
    meta: Dict[str, Any] | None = None

class AnswerOut(BaseModel):
    answer: str

class DayMonthPikIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["DAILY", "MONTHLY"] = Field(..., validation_alias=AliasChoices("type", "mode"))
    cultures: List[Dict[str, Any]]
    requestCount: int = 10
    userInterests: List[str] | None = None
    weather: str | int | None = None

class DayMonthPikOut(BaseModel):
    recommendedCultureIds: List[int]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "service": "ArtPick AI", "app_sig": APP_SIG}

@app.post("/ask", response_model=AnswerOut, dependencies=[Depends(verify)])
def ask(body: AskIn):
    if not body.cultures:
        return AnswerOut(answer="게시글 목록(cultures)이 비어 있어요. 후보 목록을 함께 보내주세요.")
    if classify_intent is None:
        raise HTTPException(500, "discriminate.classify_intent가 없습니다.")
    if recommend_json is None:
        raise HTTPException(500, "recommend.recommend_json가 없습니다.")

    data = classify_intent({"message": body.message, "meta": body.meta or {}}) or {}
    kws = data.get("keywords") or {}
    if not kws.get("when"):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        kws["when"] = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")

    user_obj = body.user.copy() if body.user else {}
    if body.userInterests:
        user_obj["interest_tags"] = ",".join([s for s in body.userInterests if s])

    result = recommend_json(
        message=body.message,
        keywords=kws,
        posts=body.cultures,
        user=user_obj,
        top_k=body.requestCount
    )

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
    if not body.cultures:
        return DayMonthPikOut(recommendedCultureIds=[])

    result = pick_ids_json(
        mode=body.type,
        posts=body.cultures,
        top_k=body.requestCount,
        interest_categories=body.userInterests or [],
        weather=body.weather
    )
    raw_ids = result.get("recommendedCultureIds", [])
    ids = []
    for x in raw_ids:
        try:
            ids.append(int(x))
        except Exception:
            continue
    return DayMonthPikOut(recommendedCultureIds=ids)
