# Sawa Food Project: Backend Architecture & Implementation Guide

This document provides a comprehensive breakdown of the FastAPI backend for the Sawa Food Project. It details what features have been implemented, how they work under the hood, and where the code resides within the directory structure.

## Overview

The backend is built as a lightweight, high-performance API designed to serve personalized meal recommendations and handle AI-driven culinary chats.

**Tech Stack:**
* **Framework**: FastAPI
* **Database**: Supabase (PostgreSQL)
* **AI Provider**: Groq (LLaMA Models)
* **Data Processing**: Pandas

---

## 1. What Has Been Implemented

The backend primarily focuses on two core domains:
1. **Personalized Feed & Recommendations**: An intelligent engine that ranks and filters Cameroonian meals based on a user's dietary restrictions, allergies, search history, sentiment (likes/dislikes), and the current time of day.
2. **CamChef AI Chatbot & Vision**: A specialized AI assistant trained exclusively on Cameroonian cuisine. It supports standard text-based chat and image-based meal identification.

---

## 2. Where & How It Was Implemented

### A. Core Setup & Database Connection
* **Location:** `database.py`, `requirements.txt`, `.env`
* **How it works:** 
  The app uses the `supabase-py` client initialized with a Service Role key. This allows the backend to perform administrative CRUD operations directly on the database (fetching dishes, updating chat sessions, checking user profiles).

### B. API Routing (The Entry Points)
* **Location:** `main.py`
* **How it works:** 
  `main.py` is the FastAPI entry point. It sets up CORS and defines the RESTful endpoints that the frontend consumes:
  * **Feed Endpoint** (`GET /api/feed/{user_id}`): Routes requests to the recommender service.
  * **Search Logging** (`POST /api/search-log`): Quickly inserts user search queries into the database to improve future recommendations.
  * **Chat Endpoints** (`POST /api/chat`, `GET /api/chat/sessions`, `GET /api/chat/history`, `DELETE /api/chat/sessions`): Manages chat sessions. It handles creating UUIDs for new sessions, persisting user/assistant messages to Supabase, and retrieving history.
  * **Image Identification** (`POST /api/identify-meal`): Accepts `multipart/form-data` (images up to specific MIME types). Validates the file, reads it into bytes, and passes it to the AI vision service. It also supports optionally saving the result directly into an active chat session.

### C. The Recommendation Engine
* **Location:** `services/recommender.py`
* **How it works:** 
  This module houses the complex logic for curating a user's feed. It uses `pandas` DataFrames to process meal data efficiently.
  * **Data Aggregation**: Conceptually fetches dishes, user profiles (allergies, conditions), recent searches, and dish sentiments (likes).
  * **Dietary Strictness**: Contains an `allergy_keywords` expander. If a user is allergic to "seafood", the system expands this to look for "shrimp", "fish", "crayfish", "lobster", etc., in the ingredients list, strictly filtering out unsafe meals.
  * **Scoring Algorithm**: Meals are scored to determine ranking:
    * *Time of Day*: Massive boost (+5000) if the meal fits the current time (e.g., Breakfast vs. Dinner).
    * *Sentiment*: Heavy penalty for disliked meals (-10000) and a boost for liked ones (+1500).
    * *Search History*: Boosts meals matching recent user queries.
    * *Health Compatibility*: Adjusts scores based on whether a meal is "safe for everyone" or matches the user's specific health conditions.
  * **Daily Rotation (Deterministic Hashing)**: Uses an MD5 hash of the `user_id`, `dish_id`, and `current_date` to apply a rotation offset. This ensures the feed changes daily but remains stable if the user refreshes the page on the same day.

### D. CamChef AI Services
* **Location:** `services/chat.py`
* **How it works:** 
  Integrates with the **Groq API** to provide ultra-fast LLM inference.
  * **System Prompt Restriction**: Uses a highly detailed `CAMEROON_SYSTEM_PROMPT`. This prompt acts as a strict boundary, forcing the LLM to act as "CamChef", identify exclusively with Cameroonian culture, speak in local terms (Pidgin, French), and politely refuse to answer questions about non-Cameroonian food.
  * **Text Chat** (`chat_with_camchef`): Uses the `llama-3.3-70b-versatile` model. It accepts the full conversation history from `main.py` so the AI retains context.
  * **Vision Model** (`identify_meal_from_image`): Uses `meta-llama/llama-4-scout-17b-16e-instruct`. It encodes the uploaded image bytes to Base64 and asks the model to extract the local name, ingredients, and nutritional data of the pictured meal.

---

## 3. The Flow of Work (System Interactions)

### Flow 1: Generating the Personalized Feed
1. The Frontend requests `/api/feed/{user_id}?mode=personalized`.
2. `main.py` forwards the request to `recommender.py`.
3. The Recommender queries Supabase for the user's profile, likes/dislikes, and search history.
4. It normalizes lists/JSON arrays of ingredients and allergies.
5. It filters out any dishes containing allergens.
6. It applies the scoring algorithm (time of day + sentiment + daily hash + health matches).
7. The top 4 highest-scoring dishes are converted to dictionaries and returned to the frontend.

### Flow 2: Identifying a Meal via Image
1. The User uploads a picture of a dish via the frontend widget.
2. The Frontend sends a `multipart/form-data` POST request to `/api/identify-meal`.
3. `main.py` validates that the image is a JPEG/PNG/WEBP/GIF.
4. The image is passed to `services/chat.py` where it is converted to Base64.
5. The Groq Vision model analyzes the image against the `CAMEROON_SYSTEM_PROMPT`.
6. Groq returns the markdown-formatted culinary details.
7. If the frontend provided a `session_id`, `main.py` inserts the user's query and the AI's response into the `chat_messages` table in Supabase.
8. The response is sent back to the frontend to be displayed.

### Flow 3: Conversing with CamChef
1. The User types a question ("How do I cook Ndolé?") and hits send.
2. The Frontend sends a JSON payload to `/api/chat` with the `user_id`, `message`, and an optional `session_id`.
3. `main.py` checks for a `session_id`. If none exists, it generates a new UUID and creates a row in the `chat_sessions` table.
4. `main.py` pulls all previous messages for that session from `chat_messages`.
5. The new message is appended to the history and sent to `chat_with_camchef` in `services/chat.py`.
6. Groq processes the history and streams back an answer.
7. `main.py` writes *both* the user's message and the assistant's reply into the `chat_messages` table.
8. The response and the `session_id` are returned to the frontend.
