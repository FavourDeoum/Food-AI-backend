import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Import components from services.recommender
from services.recommender import (
    normalize_list,
    get_allergy_keywords,
    deterministic_hash,
    get_personalized_recommendations,
    get_current_meal_type
)

class TestRecommenderImprovements(unittest.TestCase):

    def test_normalize_list(self):
        # Native list
        self.assertEqual(normalize_list(["Eggs", "Peanuts"]), ["Eggs", "Peanuts"])
        # Single string
        self.assertEqual(normalize_list("Eggs"), ["Eggs"])
        # JSON-encoded string list
        self.assertEqual(normalize_list('["Eggs", "Groundnuts"]'), ["Eggs", "Groundnuts"])
        # Comma-separated list
        self.assertEqual(normalize_list("Eggs, Groundnuts, Dairy"), ["Eggs", "Groundnuts", "Dairy"])
        # None and NaN handling
        self.assertEqual(normalize_list(None), [])
        self.assertEqual(normalize_list(float('nan')), [])

    def test_allergy_keyword_expansion(self):
        # Test plural singular expansion
        self.assertIn("egg", get_allergy_keywords("Eggs"))
        self.assertIn("eggs", get_allergy_keywords("Eggs"))
        self.assertIn("eggs", get_allergy_keywords("egg"))
        self.assertIn("egg", get_allergy_keywords("egg"))

        # Test high-level category mapping: Seafood
        seafood_kws = get_allergy_keywords("Seafood")
        self.assertIn("seafood", seafood_kws)
        self.assertIn("fish", seafood_kws)
        self.assertIn("shrimp", seafood_kws)
        self.assertIn("crayfish", seafood_kws)

        # Test high-level category mapping: Dairy
        dairy_kws = get_allergy_keywords("Dairy")
        self.assertIn("dairy", dairy_kws)
        self.assertIn("milk", dairy_kws)
        self.assertIn("butter", dairy_kws)

        # Test high-level category mapping: Gluten
        gluten_kws = get_allergy_keywords("Gluten")
        self.assertIn("gluten", gluten_kws)
        self.assertIn("wheat", gluten_kws)
        self.assertIn("flour", gluten_kws)
        self.assertIn("spaghetti", gluten_kws)

    @patch('services.recommender.supabase')
    def test_personalized_allergy_filtering(self, mock_supabase):
        # Mocking tables
        mock_dishes = [
            {"id": "dish1", "name": "Fried Potatoes and Eggs", "ingredients": "potatoes, egg", "meal_type": "Breakfast", "suitable_for": "Everyone"},
            {"id": "dish2", "name": "Boiled Plantains", "ingredients": "plantains, salt", "meal_type": "Breakfast", "suitable_for": "Everyone"},
            {"id": "dish3", "name": "Ndole", "ingredients": "bitterleaf, raw groundnuts, beef", "meal_type": "Lunch", "suitable_for": "Everyone"},
            {"id": "dish4", "name": "Roasted Fish", "ingredients": "whole mackerel, garlic, ginger", "meal_type": "Dinner", "suitable_for": "Everyone"}
        ]
        
        mock_sentiments = []
        mock_search_logs = []
        
        # User profile with "Eggs" allergy
        mock_profile = {
            "id": "test_user",
            "name": "Jane Doe",
            "food_allergies": ["Eggs"],
            "health_conditions": []
        }

        # We need custom return values depending on the table name
        def get_mock_table(table_name):
            mock_tbl = MagicMock()
            if table_name == "dishes":
                mock_tbl.select.return_value.execute.return_value.data = mock_dishes
            elif table_name == "dish_sentiments":
                mock_tbl.select.return_value.eq.return_value.execute.return_value.data = mock_sentiments
            elif table_name == "profiles":
                mock_tbl.select.return_value.eq.return_value.single.return_value.execute.return_value.data = mock_profile
            elif table_name == "search_logs":
                mock_tbl.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = mock_search_logs
            return mock_tbl

        mock_supabase.table.side_effect = get_mock_table

        # Run recommendations in personalized mode
        import asyncio
        recs = asyncio.run(get_personalized_recommendations("test_user", "personalized"))
        
        # Verify that "Fried Potatoes and Eggs" is excluded because of "Eggs" allergy -> "egg" ingredient
        rec_names = [r["name"] for r in recs]
        self.assertNotIn("Fried Potatoes and Eggs", rec_names)
        print("Personalized Allergy filtering test passed!")

    def test_daily_rotation(self):
        # We will test the deterministic scoring rotation offset across consecutive days
        user_id = "test_user_rotation"
        dish_id = "dish_abc"

        # Rotation offset for day 1
        day1_str = "20260601"
        seed_string_1 = f"{user_id}-{dish_id}-{day1_str}"
        offset1 = deterministic_hash(seed_string_1) % 1000

        # Rotation offset for day 2
        day2_str = "20260602"
        seed_string_2 = f"{user_id}-{dish_id}-{day2_str}"
        offset2 = deterministic_hash(seed_string_2) % 1000

        # Rotation offset for day 3
        day3_str = "20260603"
        seed_string_3 = f"{user_id}-{dish_id}-{day3_str}"
        offset3 = deterministic_hash(seed_string_3) % 1000

        # Verify they are deterministic, yet completely different day-to-day
        self.assertNotEqual(offset1, offset2)
        self.assertNotEqual(offset2, offset3)
        self.assertNotEqual(offset1, offset3)
        
        # Verify that they are bounded in [0, 999]
        self.assertTrue(0 <= offset1 < 1000)
        self.assertTrue(0 <= offset2 < 1000)
        self.assertTrue(0 <= offset3 < 1000)
        print(f"Daily Rotation test passed! Day 1 Offset: {offset1}, Day 2 Offset: {offset2}, Day 3 Offset: {offset3}")

if __name__ == "__main__":
    unittest.main()
