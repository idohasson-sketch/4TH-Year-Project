import os
import requests
from datetime import datetime
from supabase import create_client, Client

# ==========================================
# 1. System Configurations
# ==========================================
SUPABASE_URL = "https://pxkevqlcaiazhgqrxbsp.supabase.co"
SUPABASE_KEY = "sb_publishable_igo1VwAo9FEGetZssdbFZQ_gOBRLIi3"   

# Authorized Original Bot Token and Chat ID
TELEGRAM_BOT_TOKEN = "8809302299:AAGbuJe1Q9uifxL7ha7zHiQfZ6DYCs90W1k"  
TELEGRAM_CHAT_ID = "-1003921648414"            

print("🔄 Connecting to Supabase...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ==========================================
# 2. Advanced Telegram Notification Logic
# ==========================================
def send_telegram_alert(observations):
    """
    Sends a highly detailed real-time field report to the Telegram group,
    parsing every detail from the freshly synchronized observation data.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN.strip()}/sendMessage"
    total_records = len(observations)
    
    # Header Section
    message_lines = [
        "🚨 *A-EYE Tracker - Real-Time Field Report* 🚨",
        f"Synchronized *{total_records}* new detection(s) to the database.\n",
        "--- *Detailed Sighting Logs* ---"
    ]
    
    # Dynamically parsing details for each detected bird
    for idx, obs in enumerate(observations, 1):
        bird_name = obs.get("bird_name", "Unknown Species")
        location = obs.get("location", "Unknown Location")
        observed_at = obs.get("observed_at", "N/A")
        quantity = obs.get("quantity", 1)
        lat = obs.get("latitude", "N/A")
        lon = obs.get("longitude", "N/A")
        
        item_block = (
            f"🦉 *Sighting #{idx}: {bird_name}*\n"
            f"  • *Count:* {quantity} unit(s)\n"
            f"  • *Timestamp:* {observed_at}\n"
            f"  • *Location Name:* {location}\n"
            f"  • *Coordinates:* `{lat}, {lon}`"
        )
        message_lines.append(item_block)
        
    # Footer Section
    message_lines.append("\n📊 _All records successfully compiled and updated in Supabase._")
    
    # Merging the list into a single clean message string
    message_text = "\n".join(message_lines)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("🚀 Detailed Telegram report sent successfully to the group!")
        else:
            print(f"❌ Failed to send Telegram alert. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"⚠️ Error connecting to Telegram API: {e}")


# ==========================================
# 3. Main Processing & Synchronization
# ==========================================
def main():
    # Array structure mapping explicitly to the Database column names
    # You can append as many dynamic records as the model outputs here
    observations_to_sync = [
        {
            "bird_name": "House Sparrow",
            "location": "Jerusalem Botanical Gardens, Givat Ram",
            "observed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quantity": 1,
            "latitude": 31.7683,
            "longitude": 35.2137
        }
    ]

    print(f"⏳ Starting synchronization of {len(observations_to_sync)} observations to Supabase...")
    successful_inserts = []

    for obs in observations_to_sync:
        try:
            response = supabase.table("urban_observations").insert(obs).execute()
            if response.data:
                # Appending the successfully saved object to pass its details to Telegram
                successful_inserts.append(obs)
                print(f"  ✅ Successfully inserted: {obs['bird_name']} at {obs['location']}")
        except Exception as e:
            print(f"  ❌ Failed to insert record into Supabase: {e}")

    print(f"\n📊 Sync finished. Total records successfully saved in DB: {len(successful_inserts)}/{len(observations_to_sync)}")

    # ==========================================
    # 4. Detailed Trigger Evaluation
    # ==========================================
    if len(successful_inserts) > 0:
        print("🔔 Triggering advanced Telegram notification...")
        send_telegram_alert(successful_inserts)
    else:
        print("⏭️ No new rows were inserted. Skipping Telegram alert.")


if __name__ == "__main__":
    main()
