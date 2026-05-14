import requests

# 1. Change this to your Render URL once hosted, 
# or keep it as localhost for now
BASE_URL = "http://127.0.0.1:8000" 

# Replace this with a real ID from your Supabase Profiles table
TEST_USER_ID = "user_2pX..." 

def test_explore_feed():
    print("\n--- Testing Explore Feed ---")
    response = requests.get(f"{BASE_URL}/api/feed/{TEST_USER_ID}?mode=explore")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {len(data['data'])} dishes.")
        # Print the first dish name
        if data['data']:
            print(f"Top Recommendation: {data['data'][0]['name']}")
    else:
        print(f"Failed: {response.text}")

def test_personalized_feed():
    print("\n--- Testing Personalized Feed (Health Filtering) ---")
    response = requests.get(f"{BASE_URL}/api/feed/{TEST_USER_ID}?mode=personalized")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {len(data['data'])} health-compatible dishes.")
    else:
        print(f"Failed: {response.text}")

def test_log_search():
    print("\n--- Testing Search Logging ---")
    payload = {"user_id": TEST_USER_ID, "query": "Ndole"}
    # Note: Using params because our FastAPI endpoint uses query params
    response = requests.post(f"{BASE_URL}/api/search-log", params=payload)
    if response.status_code == 200:
        print("Success! Search logged in Supabase.")
    else:
        print(f"Failed: {response.text}")

if __name__ == "__main__":
    # Run the tests
    test_log_search()
    test_explore_feed()
    test_personalized_feed()