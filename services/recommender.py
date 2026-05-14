from datetime import datetime
import pandas as pd
from database import supabase

def get_current_meal_type():
    hour = datetime.now().hour
    if 6 <= hour < 12: return "Breakfast"
    if 12 <= hour < 16: return "Lunch"
    if 16 <= hour < 21: return "Dinner"
    return "Snack"

async def get_personalized_recommendations(user_id: str, mode: str):
    # 1. FETCH DATA IN PARALLEL (Conceptual)
    dishes_resp = supabase.table("dishes").select("*").execute()
    sentiments_resp = supabase.table("dish_sentiments").select("*").eq("user_id", user_id).execute()
    profile_resp = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
    search_resp = supabase.table("search_logs").select("query").eq("user_id", user_id).limit(5).execute()

    df_dishes = pd.DataFrame(dishes_resp.data)
    sentiments = {s['dish_id']: s['sentiment'] for s in sentiments_resp.data}
    profile = profile_resp.data
    search_queries = [s['query'].lower() for s in search_resp.data]
    current_meal = get_current_meal_type()

    # --- FILTERING ---
    # A. Remove 'unlike' dishes (Dislikes)
    unliked_ids = [did for did, sent in sentiments.items() if sent == 'unlike']
    df_dishes = df_dishes[~df_dishes['id'].isin(unliked_ids)]

    # B. Health & Allergy Filtering (For Personalized Mode)
    if mode == "personalized" and profile:
        # Filter: Must NOT contain user allergies in ingredients
        if profile.get('food_allergies'):
            allergies = set(profile['food_allergies'])
            df_dishes = df_dishes[~df_dishes['ingredients'].apply(lambda x: any(item in allergies for item in x))]

        # Filter: Suitable_for must match health_conditions
        if profile.get('health_conditions'):
            health = set(profile['health_conditions'])
            # Keep dishes that match at least one health condition
            df_dishes = df_dishes[df_dishes['suitable_for'].apply(lambda x: any(h in x for h in health))]

    # --- SCORING ---
    def calculate_score(row):
        score = 0
        # 1. Liked meals boost
        if sentiments.get(row['id']) == 'like':
            score += 100
        
        # 2. Time of day boost
        if current_meal in row['meal_type']:
            score += 50
        
        # 3. Search History Match
        for query in search_queries:
            if query in row['name'].lower() or query in row['short_description'].lower():
                score += 30
        
        return score

    df_dishes['score'] = df_dishes.apply(calculate_score, axis=1)
    
    # Sort by score descending
    ranked_dishes = df_dishes.sort_values(by='score', ascending=False)
    
    return ranked_dishes.drop(columns=['score']).to_dict(orient="records")