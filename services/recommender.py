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

    def normalize_list(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def normalize_text(value):
        return str(value).lower() if value is not None else ""

    def contains_any(text, values):
        return any(val in text for val in values if val)

    def is_safe_for_everyone(suitable_for_values):
        values = [normalize_text(item) for item in normalize_list(suitable_for_values)]
        text = " ".join(values)
        return contains_any(text, ["all", "everyone", "general", "healthy", "anyone", "no restrictions"])

    allergies = set()
    health_conditions = set()
    if profile:
        if profile.get('food_allergies'):
            allergies = {normalize_text(item) for item in normalize_list(profile['food_allergies'])}
        if profile.get('health_conditions'):
            health_conditions = {normalize_text(item) for item in normalize_list(profile['health_conditions'])}

    def has_allergy(ingredients):
        items = [normalize_text(i) for i in normalize_list(ingredients)]
        return any(allergy in item for allergy in allergies for item in items)

    def is_health_compatible(row):
        suitable_for_values = normalize_list(row.get('suitable_for'))
        suitable_text = " ".join(normalize_text(item) for item in suitable_for_values)
        if not suitable_text:
            return True
        if is_safe_for_everyone(suitable_for_values):
            return True
        if health_conditions and any(condition in suitable_text for condition in health_conditions):
            return True
        return False

    if mode == "personalized" and profile and allergies:
        df_dishes = df_dishes[~df_dishes['ingredients'].apply(has_allergy)]

    def calculate_score(row):
        score = 0
        dish_id = row['id']
        dish_name = normalize_text(row.get('name'))
        dish_desc = normalize_text(row.get('short_description'))
        meal_type = [normalize_text(item) for item in normalize_list(row.get('meal_type'))]
        suitable_for = normalize_list(row.get('suitable_for'))
        suitable_text = " ".join(normalize_text(item) for item in suitable_for)

        search_match = False
        for query in search_queries:
            if query and (query in dish_name or query in dish_desc):
                score += 10000
                search_match = True
                break

        sentiment = sentiments.get(dish_id)
        if sentiment == 'like':
            score += 1500
        elif sentiment == 'unlike':
            score -= 10000

        if any(current_meal.lower() in mt for mt in meal_type):
            score += 250

        if is_safe_for_everyone(suitable_for):
            score += 200

        if contains_any(dish_name + " " + dish_desc, ["rice", "salad", "oatmeal", "beans", "lentils", "vegetable", "fruit", "yogurt", "whole grain", "stew"]):
            score += 100

        if mode == "personalized" and profile:
            if health_conditions:
                if is_health_compatible(row):
                    score += 300
                else:
                    score -= 300

        day_seed = int(datetime.now().strftime("%Y%m%d"))
        rotation_offset = (abs(hash(f"{dish_id}-{day_seed}")) % 100) - 50
        score += rotation_offset

        if search_match:
            score += 200

        return score

    df_dishes['score'] = df_dishes.apply(calculate_score, axis=1)
    ranked_dishes = df_dishes.sort_values(by='score', ascending=False)

    if mode == "personalized":
        if len(ranked_dishes) >= 4:
            ranked_dishes = ranked_dishes.head(4)

    return ranked_dishes.drop(columns=['score']).to_dict(orient="records")