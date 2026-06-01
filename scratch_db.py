import asyncio
from database import supabase

async def inspect_db():
    print("Fetching profiles...")
    profiles_resp = supabase.table("profiles").select("*").execute()
    print(f"Found {len(profiles_resp.data)} profiles.")
    for p in profiles_resp.data:
        print(f"Profile ID: {p.get('id')}")
        print(f"  Name/Email: {p.get('display_name') or p.get('email')}")
        print(f"  Allergies: {p.get('food_allergies')} (Type: {type(p.get('food_allergies'))})")
        print(f"  Health Conditions: {p.get('health_conditions')} (Type: {type(p.get('health_conditions'))})")
        print("-" * 40)

    print("\nFetching dishes...")
    dishes_resp = supabase.table("dishes").select("*").execute()
    print(f"Found {len(dishes_resp.data)} dishes.")
    for d in dishes_resp.data[:5]:
        print(f"Dish: {d.get('name')}")
        print(f"  Ingredients: {d.get('ingredients')} (Type: {type(d.get('ingredients'))})")
        print(f"  Suitable For: {d.get('suitable_for')} (Type: {type(d.get('suitable_for'))})")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(inspect_db())
