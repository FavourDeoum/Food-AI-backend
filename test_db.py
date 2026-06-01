import sys
from database import supabase

print("Successfully loaded database client.")
try:
    # Let's fetch one profile from supabase to see what columns/types it has
    res = supabase.table("profiles").select("*").limit(1).execute()
    print("Profiles data limit 1:", res.data)
except Exception as e:
    print("Error fetching profiles:", e)

try:
    # Let's fetch one dish from supabase to see what columns/types it has
    res = supabase.table("dishes").select("*").limit(1).execute()
    print("Dishes data limit 1:", res.data)
except Exception as e:
    print("Error fetching dishes:", e)
