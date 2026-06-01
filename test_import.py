print("1. Importing os & sys...", flush=True)
import os
import sys

print("2. Importing dotenv...", flush=True)
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

print("3. Loading .env...", flush=True)
load_dotenv()

print("4. Importing supabase...", flush=True)
# pyrefly: ignore [missing-import]
from supabase import create_client, Client

print("5. Creating client...", flush=True)
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
print(f"URL: {url}", flush=True)
print(f"Key length: {len(key) if key else 0}", flush=True)
supabase = create_client(url, key)

print("6. Calling supabase...", flush=True)
res = supabase.table("profiles").select("*").limit(1).execute()
print("Success!", len(res.data), flush=True)
