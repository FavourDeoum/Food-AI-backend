import sys
from datetime import datetime
from unittest.mock import MagicMock

# 1. We must mock the supabase client BEFORE importing recommender!
# To do this, we can patch supabase in the sys.modules or just import recommender
# and immediately overwrite the supabase client reference.
print("Importing recommender module...", flush=True)
import services.recommender as recommender
print("Recommender imported successfully.", flush=True)

# Replace the supabase client inside recommender with a complete mock
mock_supabase = MagicMock()
recommender.supabase = mock_supabase
print("Mock supabase client successfully injected.", flush=True)

# Let's run a test for normalize_list
print("\n--- Testing normalize_list ---", flush=True)
assert recommender.normalize_list(["Eggs", "Peanuts"]) == ["Eggs", "Peanuts"]
assert recommender.normalize_list("Eggs") == ["Eggs"]
assert recommender.normalize_list('["Eggs", "Groundnuts"]') == ["Eggs", "Groundnuts"]
assert recommender.normalize_list("Eggs, Groundnuts, Dairy") == ["Eggs", "Groundnuts", "Dairy"]
print("normalize_list assertions passed!", flush=True)

# Let's run a test for get_allergy_keywords
print("\n--- Testing get_allergy_keywords ---", flush=True)
egg_kws = recommender.get_allergy_keywords("Eggs")
assert "egg" in egg_kws
assert "eggs" in egg_kws
seafood_kws = recommender.get_allergy_keywords("Seafood")
assert "fish" in seafood_kws
assert "shrimp" in seafood_kws
assert "crayfish" in seafood_kws
print("get_allergy_keywords assertions passed!", flush=True)

# Let's mock a database call and test recommendation filtering
print("\n--- Testing recommendation filtering ---", flush=True)
mock_dishes = [
    {"id": "dish1", "name": "Fried Potatoes and Eggs", "ingredients": "potatoes, egg", "meal_type": "Breakfast", "suitable_for": "Everyone"},
    {"id": "dish2", "name": "Boiled Plantains", "ingredients": "plantains, salt", "meal_type": "Breakfast", "suitable_for": "Everyone"},
    {"id": "dish3", "name": "Ndole", "ingredients": "bitterleaf, raw groundnuts, beef", "meal_type": "Lunch", "suitable_for": "Everyone"},
    {"id": "dish4", "name": "Roasted Fish", "ingredients": "whole mackerel, garlic, ginger", "meal_type": "Dinner", "suitable_for": "Everyone"}
]

mock_profile = {
    "id": "test_user",
    "name": "Jane Doe",
    "food_allergies": ["Eggs"],
    "health_conditions": []
}

def get_mock_table(table_name):
    mock_tbl = MagicMock()
    if table_name == "dishes":
        mock_tbl.select.return_value.execute.return_value.data = mock_dishes
    elif table_name == "dish_sentiments":
        mock_tbl.select.return_value.eq.return_value.execute.return_value.data = []
    elif table_name == "profiles":
        mock_tbl.select.return_value.eq.return_value.single.return_value.execute.return_value.data = mock_profile
    elif table_name == "search_logs":
        mock_tbl.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    return mock_tbl

mock_supabase.table.side_effect = get_mock_table

# Run recommendation async
import asyncio
recs = asyncio.run(recommender.get_personalized_recommendations("test_user", "personalized"))
rec_names = [r["name"] for r in recs]
print("Recommendations returned:", rec_names, flush=True)
assert "Fried Potatoes and Eggs" not in rec_names
print("Recommendation filtering allergy assertion passed!", flush=True)

# Let's run a test for daily rotation
print("\n--- Testing daily rotation ---", flush=True)
user_id = "test_user_rotation"
dish_id = "dish_abc"

offset1 = recommender.deterministic_hash(f"{user_id}-{dish_id}-20260601") % 1000
offset2 = recommender.deterministic_hash(f"{user_id}-{dish_id}-20260602") % 1000
offset3 = recommender.deterministic_hash(f"{user_id}-{dish_id}-20260603") % 1000

print(f"Day 1 Offset: {offset1}", flush=True)
print(f"Day 2 Offset: {offset2}", flush=True)
print(f"Day 3 Offset: {offset3}", flush=True)

assert offset1 != offset2
assert offset2 != offset3
assert offset1 != offset3
assert 0 <= offset1 < 1000
print("Daily rotation assertion passed!", flush=True)

print("\nALL TESTS PASSED SUCCESSFULLY!", flush=True)
