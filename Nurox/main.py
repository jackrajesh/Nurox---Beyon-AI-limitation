from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import re
import numpy as np

# ============================
# IMPORT CONFIG + DATABASE
# ============================

from config import GROQ_API_KEY_AI1, GROQ_API_KEY_AI2, MODEL_NAME
from database.connection import engine, SessionLocal
from database.models import Base, User, DebateHistory
from auth.routes import router as auth_router, get_current_user

# ============================
# CREATE DATABASE TABLES
# ============================

Base.metadata.create_all(bind=engine)

# ============================
# INITIALIZE FASTAPI APP
# ============================

app = FastAPI(title="NUROX V6.3 Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

# ============================
# REQUEST / RESPONSE MODELS
# ============================

class DebateRequest(BaseModel):
    question: str

class DebateMessage(BaseModel):
    role: str
    content: str

class DebateResponse(BaseModel):
    mode: str
    transcript: List[DebateMessage]
    deterministic: Optional[str]
    simulation: Optional[str]
    risk_alerts: Optional[str]
    final_answer: str
    authority: str
    confidence: str

# ============================
# SAFE LLM CALL
# ============================

async def call_llm(api_key, system_prompt, messages, temperature=0.3):
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "temperature": temperature,
                "max_tokens": 900,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()

    if not data.get("choices"):
        return "⚠ Model returned invalid structure."

    return data["choices"][0]["message"]["content"].strip()

# ============================
# MODE DETECTION
# ============================

def detect_mode(question: str):
    keywords = ["risk", "reward", "win rate", "break", "transaction", "slippage"]
    return "quant" if any(k in question.lower() for k in keywords) else "general"

# ============================
# DETERMINISTIC ENGINE
# ============================

def deterministic_engine(question: str):

    nums = list(map(float, re.findall(r"\d+\.?\d*", question)))

    if len(nums) < 2:
        return None, None

    risk = nums[0]
    reward = nums[1]
    transaction = nums[2] if len(nums) >= 3 else 0
    slippage = nums[3] if len(nums) >= 4 else 0

    net_win = reward - transaction - slippage
    net_loss = -risk - transaction

    denom = net_win - net_loss
    if denom == 0:
        return None, None

    p = -net_loss / denom
    ev = (p * net_win) + ((1 - p) * net_loss)

    return p, ev

# ============================
# MONTE CARLO EQUITY CURVE
# ============================

def monte_carlo_equity(win_prob, reward_ratio=0.02, risk_ratio=0.01, trades=200):

    equity_curve = []
    capital = 1

    for _ in range(trades):
        if np.random.rand() < win_prob:
            capital *= (1 + reward_ratio)
        else:
            capital *= (1 - risk_ratio)

        equity_curve.append(round(capital, 4))

    return equity_curve

# ============================
# PROMPTS
# ============================

PROMPT_BUILDER = """
You are a professional analyst.
Use emojis.
Use **bold** for key concepts.
Use structured sections.
Avoid text walls.
"""

PROMPT_AUDITOR = """
Provide a decisive final professional conclusion.
Use emojis.
Use **bold**.
Keep it clean and structured.
"""

# ============================
# DEBATE (PROTECTED + SAVED)
# ============================

from fastapi import Depends
from auth.routes import get_current_user

@app.post("/debate", response_model=DebateResponse)
async def debate(
    req: DebateRequest,
    current_user = Depends(get_current_user)
):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question empty.")

    mode = detect_mode(question)
    transcript: List[DebateMessage] = []

    builder = await call_llm(
        GROQ_API_KEY_AI1,
        PROMPT_BUILDER,
        [{"role": "user", "content": question}],
        0.3
    )

    transcript.append(DebateMessage(role="🧠 Builder", content=builder))

    deterministic = None
    simulation_block = None
    risk_alerts = None
    authority = "LLM"
    confidence = "Medium"

    if mode == "quant":
        p, ev = deterministic_engine(question)

        if p is not None:
            deterministic = f"🎯 **Break-even** = {p:.4f} ({p*100:.2f}%) | **EV** = {ev:.4f}"

            equity_curve = monte_carlo_equity(p)

            simulation_block = f"📊 **Equity Curve Generated** with {len(equity_curve)} trades."

            risk_alerts = (
                "🟢 **Stable Risk Profile**"
                if p > 0.4 else
                "🔴 **High Risk Profile**"
            )

            authority = "Deterministic + LLM"
            confidence = "High"

    final_answer = await call_llm(
        GROQ_API_KEY_AI2,
        PROMPT_AUDITOR,
        [{"role": "user", "content": builder}],
        0.3
    )

    # ============================
    # SAVE TO DATABASE
    # ============================

    db = SessionLocal()

    history = DebateHistory(
        user_id=current_user.id,
        question=question,
        final_answer=final_answer
    )

    db.add(history)
    db.commit()
    db.close()

    return DebateResponse(
        mode=mode,
        transcript=transcript,
        deterministic=deterministic,
        simulation=simulation_block,
        risk_alerts=risk_alerts,
        final_answer=final_answer,
        authority=authority,
        confidence=confidence
    )

# ============================
# USER HISTORY ENDPOINT
# ============================

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user)):

    db = SessionLocal()

    records = db.query(DebateHistory).filter(
        DebateHistory.user_id == current_user.id
    ).all()

    db.close()

    return records

# ============================
# HEALTH
# ============================

@app.get("/health")
async def health():
    return {"status": "NUROX V6.3 Running"}