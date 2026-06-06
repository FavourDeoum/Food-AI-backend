from datetime import datetime
import pandas as pd
import hashlib
from database import supabase

def deterministic_hash(string_val: str) -> int:
    """Returns a stable, 100% deterministic integer hash for any string."""
    return int(hashlib.md5(string_val.encode('utf-8')).hexdigest(), 16)

def get_allergy_keywords(allergy: str) -> set:
    """Expands generic allergy names into specific ingredient keywords and handles singular/plural."""
    alg = allergy.strip().lower()
    if not alg or alg == "none":
        return set()
        
    keywords = {alg}
    
    # Plural/singular expansion
    if alg.endswith('s'):
        keywords.add(alg[:-1])
    else:
        keywords.add(alg + 's')
        
    # Map high-level allergens to constituent ingredients
    allergen_map = {
        "egg": {"egg", "eggs", "omelette", "omelet"},
        "eggs": {"egg", "eggs", "omelette", "omelet"},
        "groundnut": {"groundnut", "groundnuts", "peanut", "peanuts"},
        "groundnuts": {"groundnut", "groundnuts", "peanut", "peanuts"},
        "seafood": {
            "seafood", "fish", "shrimp", "shrimps", "crayfish", "lobster", "lobsters", 
            "periwinkle", "periwinkles", "prawn", "prawns", "crab", "crabs", "cod", 
            "salmon", "tuna", "mackerel", "sardine", "sardines", "oyster", "oysters", 
            "clam", "clams", "mussel", "mussels", "squid", "octopus", "pescatarian"
        },
        "dairy": {"dairy", "milk", "butter", "cheese", "cream", "yogurt", "curd", "whey", "casein", "ghee"},
        "gluten": {"gluten", "wheat", "flour", "semolina", "spaghetti", "barley", "rye", "pasta", "macaroni", "noodle", "noodles", "bread", "dough"}
    }
    
    if alg in allergen_map:
        keywords.update(allergen_map[alg])
        
    return keywords

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
    
    # Handle missing profile gracefully
    try:
        profile_resp = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        profile = profile_resp.data
    except Exception:
        # User profile doesn't exist yet, use None
        profile = None
    
    search_resp = supabase.table("search_logs").select("query").eq("user_id", user_id).limit(5).execute()

    df_dishes = pd.DataFrame(dishes_resp.data)
    sentiments = {s['dish_id']: s['sentiment'] for s in sentiments_resp.data}
    search_queries = [s['query'].lower() for s in search_resp.data]
    current_meal = get_current_meal_type()

    def normalize_list(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, str):
            value_str = value.strip()
            # Handle JSON array representation
            if value_str.startswith('[') and value_str.endswith(']'):
                import json
                try:
                    parsed = json.loads(value_str)
                    if isinstance(parsed, list):
                        result = []
                        for item in parsed:
                            result.extend(normalize_list(item))
                        return result
                except Exception:
                    pass
            # Handle comma-separated lists
            if ',' in value_str:
                return [item.strip() for item in value_str.split(',') if item.strip()]
            return [value_str]
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
    bmi = None
    bmi_category = None
    if profile:
        if profile.get('food_allergies'):
            raw_allergies = normalize_list(profile['food_allergies'])
            for raw_alg in raw_allergies:
                allergies.update(get_allergy_keywords(raw_alg))
        if profile.get('health_conditions'):
            health_conditions = {normalize_text(item) for item in normalize_list(profile['health_conditions'])}
        # Compute BMI if weight and height are present
        try:
            w = float(profile.get('weight') or 0)
            h = float(profile.get('height') or 0)
            if w > 0 and h > 0:
                bmi = w / ((h / 100) ** 2)
                if bmi < 18.5:
                    bmi_category = "underweight"
                elif bmi < 25:
                    bmi_category = "normal"
                elif bmi < 30:
                    bmi_category = "overweight"
                else:
                    bmi_category = "obese"
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    def has_allergy(ingredients):
        if not allergies:
            return False
        items = [normalize_text(i) for i in normalize_list(ingredients)]
        for item in items:
            for kw in allergies:
                if kw in item:
                    return True
        return False

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

        # Meal type match gets a huge boost (+5000) so it's strictly prioritized over other attributes,
        # but allows rotation among relevant items.
        if any(current_meal.lower() in mt for mt in meal_type):
            score += 5000

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

            # BMI-based scoring
            if bmi_category:
                diet_labels = [normalize_text(item) for item in normalize_list(row.get('dietary_labels'))]
                suitable = normalize_text(" ".join([normalize_text(i) for i in normalize_list(row.get('suitable_for'))]))
                dish_name_lower = normalize_text(row.get('name'))
                dish_desc_lower = normalize_text(row.get('short_description', ''))
                combined_text = dish_name_lower + " " + dish_desc_lower + " " + suitable + " " + " ".join(diet_labels)

                if bmi_category == "underweight":
                    # Boost high-calorie, protein-rich, and nutrient-dense dishes
                    if contains_any(combined_text, ["protein", "high protein", "high-protein", "high calorie", "calorie dense", "energy", "beef", "chicken", "pork", "egg"]):
                        score += 400
                    if contains_any(combined_text, ["weight gain", "high energy", "bulking"]):
                        score += 350

                elif bmi_category in ("overweight", "obese"):
                    # Boost low-calorie, light, vegetable-rich dishes
                    if contains_any(combined_text, ["low calorie", "low-calorie", "light", "salad", "vegetable", "steamed", "grilled", "fiber", "high fiber", "low carb", "low-carb"]):
                        score += 400
                    if contains_any(combined_text, ["weight loss", "low fat", "low-fat"]):
                        score += 350
                    # Penalize fried and very rich dishes
                    if contains_any(combined_text, ["fried", "deep fried", "deep-fried", "fatty", "heavy"]):
                        score -= 300

                elif bmi_category == "normal":
                    # Slight boost for balanced/healthy dishes
                    if contains_any(combined_text, ["balanced", "healthy", "nutritious", "wholesome"]):
                        score += 150

        # Strong deterministic daily rotation offset unique to this user, dish, and day.
        day_str = datetime.now().strftime("%Y%m%d")
        seed_string = f"{user_id}-{dish_id}-{day_str}"
        rotation_offset = deterministic_hash(seed_string) % 1000  # range 0 to 999
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