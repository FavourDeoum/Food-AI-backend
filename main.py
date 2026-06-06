from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import uuid
from datetime import datetime, timezone

from database import supabase
from services.recommender import get_personalized_recommendations
from services.chat import chat_with_camchef, identify_meal_from_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════

class ProfileUpsertPayload(BaseModel):
    """Payload for creating or updating a user profile.
    Mirrors the Supabase `profiles` table columns.
    """
    id: str                                        # Clerk user_id (primary key)
    name: str
    age: int                                       = Field(..., ge=1, le=120)
    gender: str
    location: str
    weight: Optional[float]                        = Field(None, ge=20, le=500, description="Body weight in kilograms")
    height: Optional[float]                        = Field(None, ge=50, le=300, description="Height in centimetres")
    health_conditions: Optional[List[str]]         = None
    dietary_preference: Optional[str]              = None
    food_allergies: Optional[List[str]]            = None
    activity_level: Optional[str]                  = None
    meal_category: Optional[str]                   = None

    @field_validator("weight", "height", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """Accept numeric strings (e.g. from form submissions) and coerce to float."""
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Must be a numeric value, got: {v!r}")


class UserProfileResponse(BaseModel):
    """Full profile data returned to the client, with computed BMI fields."""
    id: str
    name: str
    age: int
    gender: str
    location: Optional[str]              = None
    weight: Optional[float]              = None
    height: Optional[float]              = None
    health_conditions: Optional[List[str]] = None
    dietary_preference: Optional[str]    = None
    food_allergies: Optional[List[str]]  = None
    activity_level: Optional[str]        = None
    meal_category: Optional[str]         = None
    # Computed fields
    bmi: Optional[float]                 = None
    bmi_category: Optional[str]          = None


def compute_bmi(weight_kg: Optional[float], height_cm: Optional[float]):
    """Return (bmi_value, bmi_category) or (None, None) if data is missing."""
    if not weight_kg or not height_cm or height_cm <= 0:
        return None, None
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return round(bmi, 1), category


# ══════════════════════════════════════════════════════════
# EXISTING ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/api/feed/{user_id}")
async def get_feed(user_id: str, mode: str = "explore"):
    try:
        recommendations = await get_personalized_recommendations(user_id, mode)
        return {"status": "success", "data": recommendations}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/search-log")
async def log_search(user_id: str, query: str):
    supabase.table("search_logs").insert({"user_id": user_id, "query": query}).execute()
    return {"status": "logged"}


# ══════════════════════════════════════════════════════════
# PROFILE ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/api/profile", response_model=UserProfileResponse, status_code=201)
async def upsert_profile(payload: ProfileUpsertPayload):
    """
    Create or update a user profile in Supabase.
    Accepts all standard profile fields plus the new body-metric fields:
      - weight (kg)
      - height (cm)

    These metrics are used by the recommendation engine to compute BMI
    and adjust dish scores accordingly.
    """
    try:
        row = payload.model_dump(exclude_none=False)  # keep None so DB nulls are written
        result = (
            supabase.table("profiles")
            .upsert(row, on_conflict="id")
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=500, detail="Upsert returned no data from Supabase")

        saved = result.data[0]
        bmi_val, bmi_cat = compute_bmi(saved.get("weight"), saved.get("height"))
        return UserProfileResponse(**saved, bmi=bmi_val, bmi_category=bmi_cat)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile/{user_id}", response_model=UserProfileResponse)
async def get_profile(user_id: str):
    """
    Retrieve a user profile by Clerk user_id.
    Returns all profile fields plus computed `bmi` and `bmi_category`
    derived from the stored weight and height values.
    """
    try:
        result = (
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Profile not found for user_id={user_id!r}")

        profile = result.data
        bmi_val, bmi_cat = compute_bmi(profile.get("weight"), profile.get("height"))
        return UserProfileResponse(**profile, bmi=bmi_val, bmi_category=bmi_cat)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════
# CHAT ENDPOINTS
# ══════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatPayload(BaseModel):
    user_id: str
    session_id: Optional[str] = None   # omit to start a new session
    message: str                        # only the latest user message


@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    """
    Send a message to CamChef. Pass session_id to continue an existing
    conversation, or omit it to start a fresh one.

    The endpoint:
      1. Creates a session row if session_id is not provided.
      2. Loads the full message history for that session from the DB.
      3. Appends the new user message and calls the model.
      4. Saves both the user message and assistant reply to the DB.
      5. Returns the reply and the session_id.
    """
    try:
        session_id = payload.session_id

        # ── 1. Create session if needed ───────────────────────────────────────
        if not session_id:
            session_id = str(uuid.uuid4())
            supabase.table("chat_sessions").insert({
                "id": session_id,
                "user_id": payload.user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()

        # ── 2. Load history ───────────────────────────────────────────────────
        history_result = (
            supabase.table("chat_messages")
            .select("role, content")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        history = history_result.data or []   # [{"role": ..., "content": ...}]

        # ── 3. Append new user message and call the model ─────────────────────
        history.append({"role": "user", "content": payload.message})
        reply = chat_with_camchef(history)

        # ── 4. Persist user message + assistant reply ─────────────────────────
        now = datetime.now(timezone.utc).isoformat()
        supabase.table("chat_messages").insert([
            {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "role": "user",
                "content": payload.message,
                "created_at": now,
            },
            {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "role": "assistant",
                "content": reply,
                "created_at": now,
            },
        ]).execute()

        # Update session timestamp
        supabase.table("chat_sessions").update({
            "updated_at": now,
        }).eq("id", session_id).execute()

        return {"status": "success", "session_id": session_id, "reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """
    Return all chat sessions for a user, newest first.
    Use this to populate a 'chat history' list in the frontend.
    """
    try:
        result = (
            supabase.table("chat_sessions")
            .select("id, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return {"status": "success", "sessions": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history/{session_id}")
async def get_session_history(session_id: str):
    """
    Return all messages for a specific session, oldest first.
    Use this to restore a conversation when the user reopens it.
    """
    try:
        result = (
            supabase.table("chat_messages")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return {"status": "success", "messages": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    try:
        supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════
# MEAL IMAGE IDENTIFICATION ENDPOINT
# ══════════════════════════════════════════════════════════

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@app.post("/api/identify-meal")
async def identify_meal_endpoint(
    image: UploadFile = File(...),
    question: Optional[str] = Form(default=""),
    user_id: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),  # saves result to chat session if provided
):
    """
    Upload a meal image to get its Cameroonian name and full details.
    If session_id is provided, the result is saved to that chat session
    so the user can reference it later.

    Accepts multipart/form-data:
      - image      : JPEG / PNG / WEBP / GIF
      - question   : (optional) specific question about the meal
      - user_id    : (optional) user identifier
      - session_id : (optional) attach result to an existing chat session
    """
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{image.content_type}'. Use JPEG, PNG, WEBP, or GIF.",
        )

    try:
        image_bytes = await image.read()
        reply = identify_meal_from_image(image_bytes, image.content_type, question or "")

        # Optionally persist to chat history
        if session_id and user_id:
            now = datetime.now(timezone.utc).isoformat()
            user_content = question.strip() if question and question.strip() else "🖼️ [Uploaded a meal image for identification]"
            supabase.table("chat_messages").insert([
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "role": "user",
                    "content": user_content,
                    "created_at": now,
                },
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "role": "assistant",
                    "content": reply,
                    "created_at": now,
                },
            ]).execute()
            supabase.table("chat_sessions").update({"updated_at": now}).eq("id", session_id).execute()

        return {"status": "success", "reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
