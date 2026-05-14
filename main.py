from fastapi import FastAPI, BackgroundTasks
from database import supabase
from services.recommender import get_personalized_recommendations

app = FastAPI()

# Endpoint for the Frontend to get the feed
@app.get("/api/feed/{user_id}")
async def get_feed(user_id: str, mode: str = "explore"):
    try:
        recommendations = await get_personalized_recommendations(user_id, mode)
        return {"status": "success", "data": recommendations}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Endpoint to log searches
@app.post("/api/search-log")
async def log_search(user_id: str, query: str):
    supabase.table("search_logs").insert({"user_id": user_id, "query": query}).execute()
    return {"status": "logged"}