import os
import requests
import time
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
HOME_DIR = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')
# iNaturalist
TARGET_SPECIES = \
    {
    "House_Sparrow": 13858,
    "Feral_Pigeon": 3017,
    "Rose_ringed_Parakeet": 18874,
    "Hooded_Crow": 133271
}
# Teams
TIERS = \
    {
    "T1-1000": 1000,
    "T2-150": 150,
    "T3-50": 50
}
BATCH_SIZE = 50


def download_verified_dataset():
    print(f"[*] Initializing Milestone 2 Data Acquisition Pipeline...")
    print(f"[*] Target Directory: {DATASET_DIR}\n")

    if os.path.exists(DATASET_DIR):
        print("[*] Cleaning up old dataset directory...")
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)
    session = requests.Session()

    #Download 
    for species_name, taxon_id in TARGET_SPECIES.items():
        process_class_download(session, species_name, taxon_id)
    # 2. Download "Other" 
    excluded_ids = ",".join(map(str, TARGET_SPECIES.values()))
    process_class_download(session, "Other", taxon_id=3, exclude_ids=excluded_ids)


def process_class_download(session, class_name, taxon_id, exclude_ids=None):
    class_folder = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)
    print(f"\n==================================================")
    print(f"[+] Starting fetch for class: {class_name}")
    print(f"==================================================")
    page = 1
    for tier_name, limit in TIERS.items():
        tier_folder = os.path.join(class_folder, tier_name)
        os.makedirs(tier_folder, exist_ok=True)
        print(f"\n[->] Filling tier {tier_name} ({limit} images)...")
        downloaded_in_tier = 0
        while downloaded_in_tier < limit:
            url = "https://api.inaturalist.org/v1/observations"
            params = {
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "per_page": BATCH_SIZE,
                "page": page,
                "photos": "true"
            }
            if exclude_ids:
                params["without_taxon_id"] = exclude_ids
            try:
                response = session.get(url, params=params, timeout=15)
                if response.status_code != 200:
                    print(f"[X] API Error {response.status_code}, retrying...")
                    time.sleep(2)
                    continue

                data = response.json()
                results = data.get("results", [])
                if not results:
                    print(f"[!] No more verified images available for {class_name}.")
                    break
                for obs in results:
                    if downloaded_in_tier >= limit:
                        break
                    photos = obs.get("observation_photos", [])
                    if not photos:
                        continue
                    photo_url = photos[0].get("photo", {}).get("url", "")
                    if not photo_url:
                        continue
                    if tier_name == "T3-50":
                        photo_url = photo_url.replace("square", "original").replace("small", "original")
                    else:
                        photo_url = photo_url.replace("square", "medium").replace("small", "medium")
                    file_name = f"{class_name.lower()}_{obs['id']}.jpg"
                    file_path = os.path.join(tier_folder, file_name)
                    if os.path.exists(file_path):
                        downloaded_in_tier += 1
                        continue

                    try:
                        img_data = session.get(photo_url, timeout=10).content
                        with open(file_path, 'wb') as handler:
                            handler.write(img_data)
                        downloaded_in_tier += 1
                    except Exception:
                        if "original" in photo_url:
                            photo_url = photo_url.replace("original", "medium")
                            img_data = session.get(photo_url, timeout=10).content
                            with open(file_path, 'wb') as handler:
                                handler.write(img_data)
                            downloaded_in_tier += 1
                        else:
                            continue

                    if downloaded_in_tier % 20 == 0 or downloaded_in_tier == limit:
                        print(f"    -> {tier_name} Progress: {downloaded_in_tier}/{limit} successfully saved.")

                    time.sleep(0.05)

                page += 1

            except Exception as e:
                print(f"[X] Connection issue encountered: {e}")
                time.sleep(2)

        print(f"[V] Tier {tier_name} complete for {class_name}.")


if __name__ == "__main__":
    download_verified_dataset()
