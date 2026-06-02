import os
import requests
import time
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- CONFIGURATION ---
HOME_DIR = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME_DIR, 'Downloads', 'birds_dataset')

# UPDATED: Using the definitive master Taxon IDs to prevent database overlaps
TARGET_SPECIES = {
    "House Sparrow": 13858,
    "Feral Pigeon": 3017
}

IMAGES_PER_CLASS = 150
BATCH_SIZE = 30


def download_verified_dataset():
    print(f"[*] Initializing Verified Data Acquisition Pipeline...")
    print(f"[*] Target Directory: {DATASET_DIR}")

    # Reset dataset directory if it contains corrupted/wrong classes
    if os.path.exists(DATASET_DIR):
        print("[*] Cleaning up old dataset directory...")
        shutil.rmtree(DATASET_DIR)

    os.makedirs(DATASET_DIR)

    session = requests.Session()

    for species_name, taxon_id in TARGET_SPECIES.items():
        species_folder = os.path.join(DATASET_DIR, species_name)
        if not os.path.exists(species_folder):
            os.makedirs(species_folder)

        print(f"\n[+] Fetching verified imagery for: {species_name} (Taxon ID: {taxon_id})")

        downloaded_count = 0
        page = 1

        while downloaded_count < IMAGES_PER_CLASS:
            url = f"https://api.inaturalist.org/v1/observations"
            params = {
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "per_page": BATCH_SIZE,
                "page": page,
                "photos": "true"
            }

            try:
                response = session.get(url, params=params, timeout=15)
                if response.status_code != 200:
                    print(f"[X] API Error status {response.status_code}, retrying...")
                    time.sleep(2)
                    continue

                data = response.json()
                results = data.get("results", [])

                if not results:
                    print("[!] No more verified images available on iNaturalist for this species.")
                    break

                for obs in results:
                    if downloaded_count >= IMAGES_PER_CLASS:
                        break

                    photos = obs.get("observation_photos", [])
                    if not photos:
                        continue

                    photo_url = photos[0].get("photo", {}).get("url", "")
                    if not photo_url:
                        continue

                    photo_url = photo_url.replace("square", "medium").replace("small", "medium")

                    file_name = f"{species_name.lower().replace(' ', '_')}_{obs['id']}.jpg"
                    file_path = os.path.join(species_folder, file_name)

                    if os.path.exists(file_path):
                        downloaded_count += 1
                        continue

                    img_data = session.get(photo_url, timeout=10).content
                    with open(file_path, 'wb') as handler:
                        handler.write(img_data)

                    downloaded_count += 1
                    if downloaded_count % 10 == 0 or downloaded_count == IMAGES_PER_CLASS:
                        print(f"    -> Progress: Captured {downloaded_count}/{IMAGES_PER_CLASS} images.")

                    time.sleep(0.1)

                page += 1

            except Exception as e:
                print(f"[X] Connection issue or timeout encountered: {e}")
                time.sleep(2)

        print(f"[V] Successfully generated clean dataset for {species_name} ({downloaded_count} images).")


if __name__ == "__main__":
    download_verified_dataset()
