"""
services/ml_recommender.py
==========================
ML-powered diet classification service for the Cameroonian Food AI Platform.

This module loads the trained RandomForest model and exposes a single
public function `get_ml_diet_context()` that converts a Supabase user
profile into:
  - A diet label  (Low_Carb | Low_Sodium | Balanced)
  - Dish matching tags  (suitable_for / dietary_labels)
  - A human-readable recommendation reason
  - An allergy-aware filter list

The output feeds directly into the scoring function in recommender.py
to re-rank the Cameroonian dish catalogue for each user.

INTEGRATION STEPS
-----------------
1. Place diet_classifier.pkl and model_metadata.json next to this file
   (or adjust MODEL_DIR below).
2. Replace the scoring section in services/recommender.py with a call
   to this module (see example at the bottom of this file).
3. Run:  pip install scikit-learn joblib pandas numpy
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE)          # same directory as this file

MODEL_PATH    = os.path.join(MODEL_DIR, "diet_classifier.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ── Encoding maps (must be identical to training time) ────────────────────────
HEALTH_CONDITION_MAP: dict[str, int] = {
    "none": 0,
    "ulcer": 1,
    "weight loss": 2,
    "weight gain": 3,
    "allergies": 4,
    "bp": 5,
    "hypertension": 5,
    "blood pressure": 5,
    "diabetes": 6,
    "high sugar": 6,
}

ACTIVITY_MAP: dict[str, int] = {
    "low activity": 0,
    "moderate activity": 1,
    "high activity": 2,
}

GENDER_MAP: dict[str, int] = {"male": 0, "female": 1}
SEVERITY_MAP: dict[str, int] = {"mild": 0, "moderate": 1, "severe": 2}


# ── Dish-tag routing table ─────────────────────────────────────────────────────
# Maps each diet label to the Supabase dish fields we use for scoring.
DIET_TO_DISH_TAGS: dict[str, dict[str, Any]] = {
    "Low_Carb": {
        "dietary_labels": ["Low-Carb", "Keto-Friendly", "High-Protein", "Pescatarian"],
        "suitable_for": [
            "Blood Sugar Control", "Weight Management",
            "Muscle Building", "Weight Loss",
        ],
        "preferred_categories": ["Protein", "Soup", "Light"],
        "avoid_categories": ["Snack", "Breakfast"],
        "calorie_preference": "moderate",   # 300-600 kcal
        "boost_multiplier": 3.0,
    },
    "Low_Sodium": {
        "dietary_labels": [
            "Dairy-Free", "Low-Calorie", "High-Fiber",
            "Gluten-Free", "Vegetarian",
        ],
        "suitable_for": [
            "Heart Health", "Digestive Health",
            "Sustained Energy", "Immune Support",
        ],
        "preferred_categories": ["Soup", "Light", "Traditional"],
        "avoid_categories": [],
        "calorie_preference": "low",        # <400 kcal
        "boost_multiplier": 3.0,
    },
    "Balanced": {
        "dietary_labels": [
            "High-Protein", "High-Fiber", "Gluten-Free",
            "Vegan", "Dairy-Free",
        ],
        "suitable_for": [
            "General Health", "Energy Boost",
            "Sustained Energy", "Comfort Eating",
            "Anemia Prevention", "Quick Energy",
        ],
        "preferred_categories": ["Traditional", "Protein", "Breakfast", "Snack"],
        "avoid_categories": [],
        "calorie_preference": "normal",     # any
        "boost_multiplier": 2.0,
    },
}


# ── Allergy keyword expansion ──────────────────────────────────────────────────
_ALLERGY_EXPAND: dict[str, set[str]] = {
    "groundnuts": {"groundnut", "groundnuts", "peanut", "peanuts"},
    "seafood": {
        "seafood", "fish", "shrimp", "shrimps", "crayfish",
        "periwinkle", "periwinkles", "prawn", "prawns",
        "crab", "crabs", "lobster", "mackerel", "sardine",
        "oyster", "oysters", "clam", "mussel", "squid",
    },
    "dairy": {"dairy", "milk", "butter", "cheese", "cream", "yogurt", "ghee"},
    "gluten": {
        "gluten", "wheat", "flour", "semolina", "spaghetti",
        "barley", "rye", "pasta", "macaroni", "noodle", "bread",
    },
    "eggs": {"egg", "eggs", "omelette", "omelet"},
}


def expand_allergy_keywords(raw_allergies: list[str]) -> set[str]:
    """Return a flat set of ingredient keywords to block for the user."""
    blocked: set[str] = set()
    for allergy in raw_allergies:
        key = allergy.strip().lower()
        if key in ("none", ""):
            continue
        expanded = _ALLERGY_EXPAND.get(key)
        if expanded:
            blocked.update(expanded)
        else:
            blocked.add(key)
            if key.endswith("s"):
                blocked.add(key[:-1])
            else:
                blocked.add(key + "s")
    return blocked


# ── Model loading (cached so it only happens once per worker) ─────────────────
@lru_cache(maxsize=1)
def _load_model():
    """Load the pickled RandomForest model exactly once."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the training notebook first to generate diet_classifier.pkl."
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Diet classifier loaded from %s", MODEL_PATH)
    return model


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH) as fh:
        return json.load(fh)


# ── Feature extraction ────────────────────────────────────────────────────────
def _profile_to_feature_vector(profile: dict) -> list[float]:
    """
    Convert a Supabase ``profiles`` row into the 12-element feature vector
    expected by the newly trained model (matches train_model.py).
    """
    # Age & Gender normalization
    age = float(profile.get('age') or 30)
    gender_str = str(profile.get('gender') or 'Male').lower()
    gender_num = 1 if gender_str == 'female' else (2 if gender_str == 'other' else 0)
    
    # Dynamic BMI Calculation from new metric fields
    try:
        weight = float(profile.get("weight") or 70.0)
        height = float(profile.get("height") or 170.0)
        height_m = height / 100.0
        if height_m > 0:
            bmi = weight / (height_m ** 2)
        else:
            bmi = 22.0
    except (ValueError, TypeError):
        bmi = 22.0

    # Extract health conditions
    raw_conds = profile.get("health_conditions") or ["None"]
    if isinstance(raw_conds, str):
        try:
            raw_conds = json.loads(raw_conds)
        except Exception:
            raw_conds = [raw_conds]
    conds_lower = [c.strip().lower() for c in raw_conds if c]
    condition = " ".join(conds_lower)
    
    # 1-Hot encoding user health and diet goals
    has_high_bp = 1 if any(x in condition for x in ['bp', 'hypertension', 'blood pressure']) else 0
    has_diabetes = 1 if any(x in condition for x in ['diabetes', 'diabetic', 'high sugar']) else 0
    has_ulcer = 1 if 'ulcer' in condition else 0
    has_yeast = 1 if 'yeast' in condition else 0
    has_allergies = 1 if any(x in condition for x in ['allergies', 'allergy']) else 0
    weight_loss = 1 if 'weight loss' in condition else 0
    weight_gain = 1 if 'weight gain' in condition else 0
    is_healthy = 1 if any(x in condition for x in ['none', 'healthy', 'normal']) else 0

    # Automatically flag weight gain (obesity proxy) for the model if BMI >= 30
    if bmi >= 30.0:
        weight_gain = 1

    # Activity mapping
    activity_str = str(profile.get('activity_level') or 'Moderate activity').lower()
    if 'low' in activity_str:
        activity_score = 0
    elif 'high' in activity_str:
        activity_score = 2
    else:
        activity_score = 1
        
    return [
        float(age), float(gender_num), float(bmi), float(has_high_bp), float(has_diabetes), float(has_ulcer), 
        float(has_yeast), float(has_allergies), float(weight_loss), float(weight_gain), float(is_healthy), float(activity_score)
    ]

# ── Recommendation reason strings ─────────────────────────────────────────────
def _build_reason(diet_label: str, profile: dict) -> str:
    conds = [
        c for c in (profile.get("health_conditions") or [])
        if c.strip().lower() not in ("none", "")
    ]
    cond_str = ", ".join(conds) if conds else "your health profile"

    reasons = {
        "Low_Carb": (
            f"Based on {cond_str}, low-carbohydrate Cameroonian dishes "
            "help regulate blood sugar, support weight management, and "
            "provide sustained energy without glucose spikes."
        ),
        "Low_Sodium": (
            f"Given {cond_str}, heart-friendly dishes with naturally "
            "lower sodium keep blood pressure in check while still "
            "delivering rich Cameroonian flavours."
        ),
        "Balanced": (
            "A balanced selection of traditional Cameroonian meals "
            "gives you the right mix of protein, fibre, and complex "
            "carbohydrates to fuel your day."
        ),
    }
    return reasons.get(diet_label, "Personalised recommendation based on your profile.")


# ── Public API ────────────────────────────────────────────────────────────────
def get_ml_diet_context(profile: dict) -> dict:
    """
    Main entry point.  Call this from ``services/recommender.py``.

    Parameters
    ----------
    profile : dict
        A row from the Supabase ``profiles`` table, e.g.::

            {
              "id": "clerk_user_xxx",
              "name": "Mary",
              "age": 62,
              "gender": "Female",
              "health_conditions": ["Hypertension"],
              "dietary_preference": "Balanced diet",
              "food_allergies": ["Seafood"],
              "activity_level": "Moderate activity",
              "meal_category": "Dinner",
            }

    Returns
    -------
    dict with keys:
        diet_label        : str  — "Low_Carb" | "Low_Sodium" | "Balanced"
        confidence        : float — model probability for top class (0-1)
        probability_breakdown : dict[str, float]
        dish_tags         : dict  — dietary_labels, suitable_for, …
        recommendation_reason : str
        blocked_ingredients : set[str] — ingredients to exclude (allergies)
    """
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        logger.warning("ML model unavailable: %s — falling back to rule-based.", exc)
        return _rule_based_fallback(profile)

    features = _profile_to_feature_vector(profile)
    feat_arr = np.array([features])

    diet_label: str = model.predict(feat_arr)[0]
    probabilities   = model.predict_proba(feat_arr)[0]
    confidence      = float(max(probabilities))

    # Allergy blocking
    raw_allergies = profile.get("food_allergies") or []
    if isinstance(raw_allergies, str):
        try:
            raw_allergies = json.loads(raw_allergies)
        except json.JSONDecodeError:
            raw_allergies = [raw_allergies]
    blocked = expand_allergy_keywords(raw_allergies)

    return {
        "diet_label": diet_label,
        "confidence": confidence,
        "probability_breakdown": {
            cls: float(p)
            for cls, p in zip(model.classes_, probabilities)
        },
        "dish_tags": DIET_TO_DISH_TAGS.get(diet_label, DIET_TO_DISH_TAGS["Balanced"]),
        "recommendation_reason": _build_reason(diet_label, profile),
        "blocked_ingredients": blocked,
    }


def _rule_based_fallback(profile: dict) -> dict:
    """
    Simple deterministic fallback used when the model file is missing.
    Mirrors the decision-tree logic learned from the dataset:
      Diabetes / High Sugar  -> Low_Carb
      Hypertension / BP      -> Low_Sodium
      Everything else        -> Balanced
    """
    conds = [
        c.strip().lower()
        for c in (profile.get("health_conditions") or [])
    ]
    if any(c in ("diabetes", "high sugar") for c in conds):
        label = "Low_Carb"
    elif any(c in ("hypertension", "bp", "blood pressure") for c in conds):
        label = "Low_Sodium"
    else:
        label = "Balanced"

    raw_allergies = profile.get("food_allergies") or []
    if isinstance(raw_allergies, str):
        try:
            raw_allergies = json.loads(raw_allergies)
        except json.JSONDecodeError:
            raw_allergies = [raw_allergies]

    return {
        "diet_label": label,
        "confidence": 1.0,
        "probability_breakdown": {label: 1.0},
        "dish_tags": DIET_TO_DISH_TAGS.get(label, DIET_TO_DISH_TAGS["Balanced"]),
        "recommendation_reason": _build_reason(label, profile),
        "blocked_ingredients": expand_allergy_keywords(raw_allergies),
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    samples = [
        {"name": "John",  "age": 55, "gender": "Male",   "health_conditions": ["Diabetes"],          "activity_level": "Low activity",      "food_allergies": ["Groundnuts"]},
        {"name": "Mary",  "age": 62, "gender": "Female", "health_conditions": ["Hypertension", "BP"],"activity_level": "Moderate activity",  "food_allergies": ["None"]},
        {"name": "Paul",  "age": 28, "gender": "Male",   "health_conditions": ["None"],              "activity_level": "High activity",      "food_allergies": ["None"]},
        {"name": "Grace", "age": 45, "gender": "Female", "health_conditions": ["Weight Loss"],        "activity_level": "Moderate activity",  "food_allergies": ["Seafood", "Gluten"]},
        {"name": "Peter", "age": 38, "gender": "Male",   "health_conditions": ["Diabetes","Weight Gain"],"activity_level": "Low activity",  "food_allergies": ["None"]},
    ]

    for p in samples:
        ctx = get_ml_diet_context(p)
        print(f"\n{'='*60}")
        print(f"Patient : {p['name']} | Age {p['age']} | {p['health_conditions']}")
        print(f"Diet    : {ctx['diet_label']}  (confidence {ctx['confidence']:.0%})")
        print(f"Tags    : {ctx['dish_tags']['suitable_for'][:3]}")
        print(f"Blocked : {sorted(ctx['blocked_ingredients'])[:5]}")
        print(f"Reason  : {ctx['recommendation_reason'][:100]}...")