import os
import requests
import re
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- Configurations ---
COUNTRY_CODE = "IL"
EBIRD_API_KEY = "kuj19arnk19s"  

# Fetch variables from GitHub Environment Secrets
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print(f"❌ ERROR: Missing credentials! URL: {bool(SUPABASE_URL)}, KEY: {bool(SUPABASE_KEY)}")
    raise ValueError("Missing Supabase Environment Variables!")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Watchlist for Urban Birds
WATCHLIST = [
    "Rock Pigeon",
    "Feral Pigeon",
    "House Sparrow",
    "Spanish Sparrow",
    "Dead Sea Sparrow",
    "Common Wood-Pigeon"
]

def save_to_supabase(bird_name, location, obs_dt, qty, lat, lng):
    try:
        if len(obs_dt) == 10:
            obs_dt += " 00:00"
            
        # Overriding location metrics to force database alignment
        forced_location = "Jerusalem Botanical Gardens, Givat Ram"
        forced_lat = 31.768
        forced_lng = 35.201
            
        data = {
            "bird_name": bird_name,
            "location": forced_location,
            "observed_at": obs_dt,
            "quantity": int(qty),
            "latitude": float(forced_lat),
            "longitude": float(forced_lng)
        }
        
        # Explicit execution to capture errors in the log
        supabase.table("urban_observations").insert(data).execute()
        return True
    except Exception as e:
        print(f"❌ DB INSERT ERROR for {bird_name}: {e}")
        return False

def check_historic_watchlist(days=90):
    print(f"🔄 Fetching urban birds data for the last {days} days and syncing with Supabase...")
    headers = {"X-eBirdApiToken": EBIRD_API_KEY}
    all_matches = []
    seen_records = set()
    
    for chunk in range(0, days, 30):
        chunk_days = min(30, days - chunk)
        end_date = datetime.now() - timedelta(days=chunk)
        
        if chunk == 0:
            url = f"https://api.ebird.org/v2/data/obs/{COUNTRY_CODE}/recent"
            params = {"back": chunk_days, "sppLocale": "en"}
        else:
            target_date_str = end_date.strftime("%Y/%m/%d")
            url = f"https://api.ebird.org/v2/data/obs/{COUNTRY_CODE}/historic/{target_date_str}"
            params = {"back": chunk_days, "sppLocale": "en"}
            
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 400:
                continue
            response.raise_for_status()
            observations = response.json()
            
            for obs in observations:
                bird_name = obs.get("comName")
                if bird_name in WATCHLIST or (bird_name and any(w in bird_name.lower() for w in ["pigeon", "sparrow", "dove"])):
                    record_key = (bird_name, obs.get("locName"), obs.get("obsDt"))
                    if record_key not in seen_records:
                        seen_records.add(record_key)
                        all_matches.append(obs)
                        
        except Exception as e:
            print(f"⚠️ Fetch Issue: {e}")
            break

    if not all_matches:
        print(f"⚠️ No matching urban birds found in the last {days} days.")
        return

    all_matches.sort(key=lambda x: x.get("obsDt", ""), reverse=True)

    print(f"✅ Found {len(all_matches)} urban bird observations. Syncing to DB...\n")
    
    for obs in all_matches:
        bird_name = obs.get("comName", "Unknown")
        obs_dt = obs.get("obsDt", "Unknown")
        how_many = obs.get("howMany", 1)
        lat = obs.get("lat")
        lng = obs.get("lng")
        
        save_to_supabase(bird_name, obs.get("locName", "Unknown"), obs_dt, how_many, lat, lng)
        print(f"🕊️ Processing: {bird_name:<25} | Forced to Jerusalem | {obs_dt}")

    print("\n📊 Sync finished.")

if __name__ == '__main__':
    check_historic_watchlist(days=90)
