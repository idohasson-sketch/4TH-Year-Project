"""
===============================================================================
A-EYE TRACKER — Regional Wildlife Survey & Pre-Deployment Reconnaissance Tool
===============================================================================
Purpose:
Conducts preliminary spatial reconnaissance and species distribution analysis
within a defined geographic perimeter (coordinates and radius) over the past 90
days using the global eBird observational dataset.

This tool evaluates field activity and species presence prior to physical camera
trap deployment, ensuring adequate target species density, aiding TinyML
architecture planning, and validating category alignment with the local habitat.
===============================================================================
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from supabase import create_client, Client

# --- Embedded System Configurations ---
EBIRD_API_KEY = "kuj19arnk19s"
SUPABASE_URL = "https://pxkevqlcaiazhgqrxbsp.supabase.co"
SUPABASE_KEY = "sb_publishable_igo1VwAo9FEGetZssdbFZQ_gOBRLIi3"


def get_supabase_client() -> Optional[Client]:
    """Initializes and returns a Supabase client using embedded credentials."""
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"⚠️ Failed to initialize Supabase client: {e}")
        return None


def fetch_regional_observations(
    target_species: List[str],
    lat: float,
    lng: float,
    radius_km: int = 10,
    days: int = 90,
    sync_to_db: bool = False,
    api_key: str = EBIRD_API_KEY
) -> List[Dict]:
    """
    Fetches bird observations for specified target species within a given coordinate radius
    over the past X days using the eBird 2.0 API.

    :param target_species: List of common bird names to query (e.g., ['House Sparrow', 'Feral Pigeon']).
    :param lat: Latitude of the targeted deployment zone.
    :param lng: Longitude of the targeted deployment zone.
    :param radius_km: Search radius in kilometers (max 50km per eBird API spec).
    :param days: Lookback period in days (default: 90).
    :param sync_to_db: Flag to persist matched observations into Supabase.
    :param api_key: eBird API access token.
    :return: List of structured observation dictionaries.
    """
    print(f"📡 [Survey Initialized] Area: ({lat}, {lng}) | Radius: {radius_km} km | Lookback: {days} days")
    print(f"🎯 Target Species ({len(target_species)}): {', '.join(target_species)}")

    headers = {"X-eBirdApiToken": api_key}
    normalized_targets = [s.strip().lower() for s in target_species]
    
    all_matched_observations = []
    seen_records = set()
    
    # eBird API partitions historical queries into 30-day lookup chunks
    for chunk in range(0, days, 30):
        chunk_days = min(30, days - chunk)
        end_date = datetime.now() - timedelta(days=chunk)
        
        if chunk == 0:
            url = "https://api.ebird.org/v2/data/obs/geo/recent"
            params = {
                "lat": lat,
                "lng": lng,
                "dist": radius_km,
                "back": chunk_days,
                "sppLocale": "en"
            }
        else:
            target_date_str = end_date.strftime("%Y/%m/%d")
            url = f"https://api.ebird.org/v2/data/obs/geo/historic/{target_date_str}"
            params = {
                "lat": lat,
                "lng": lng,
                "dist": radius_km,
                "back": chunk_days,
                "sppLocale": "en"
            }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 400:
                continue
            response.raise_for_status()
            records = response.json()
            
            for rec in records:
                com_name = rec.get("comName", "")
                is_match = any(
                    target == com_name.lower() or target in com_name.lower()
                    for target in normalized_targets
                )
                
                if is_match:
                    record_key = (com_name, rec.get("locId"), rec.get("obsDt"))
                    if record_key not in seen_records:
                        seen_records.add(record_key)
                        
                        obs_dt = rec.get("obsDt", "")
                        if len(obs_dt) == 10:
                            obs_dt += " 00:00"
                            
                        parsed_record = {
                            "bird_name": com_name,
                            "scientific_name": rec.get("sciName", ""),
                            "location_name": rec.get("locName", "Unknown Location"),
                            "observed_at": obs_dt,
                            "quantity": int(rec.get("howMany", 1)),
                            "latitude": float(rec.get("lat", lat)),
                            "longitude": float(rec.get("lng", lng)),
                            "obs_valid": rec.get("obsValid", True),
                            "loc_id": rec.get("locId")
                        }
                        all_matched_observations.append(parsed_record)
                        
        except Exception as e:
            print(f"⚠️ Warning during query for date chunk offset {chunk}d: {e}")
            break

    # Sort observations chronologically (most recent first)
    all_matched_observations.sort(key=lambda x: x.get("observed_at", ""), reverse=True)
    print(f"✅ Reconnaissance complete: Retrieved {len(all_matched_observations)} verified records.")

    # Optional synchronization with cloud database
    if sync_to_db and all_matched_observations:
        supabase = get_supabase_client()
        if supabase:
            print(f"💾 Synchronizing {len(all_matched_observations)} records to Supabase 'urban_observations' table...")
            for obs in all_matched_observations:
                try:
                    db_payload = {
                        "bird_name": obs["bird_name"],
                        "location": obs["location_name"],
                        "observed_at": obs["observed_at"],
                        "quantity": obs["quantity"],
                        "latitude": obs["latitude"],
                        "longitude": obs["longitude"]
                    }
                    supabase.table("urban_observations").insert(db_payload).execute()
                except Exception as e:
                    print(f"❌ DB Insertion Error for {obs['bird_name']}: {e}")
        else:
            print("⚠️ Supabase client unavailable. Skipped database synchronization.")

    return all_matched_observations
