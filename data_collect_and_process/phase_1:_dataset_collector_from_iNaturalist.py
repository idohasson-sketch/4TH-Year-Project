"""
===============================================================================
A-EYE TRACKER — Research-Grade Dataset Mining & Tier Acquisition Pipeline
===============================================================================
Purpose:
Automates the retrieval and organization of verified wildlife imagery from the
global iNaturalist API (Research Grade quality) across target bird species and an
explicit 'Other' baseline class.

Key Responsibilities & Architecture:
1. Multi-Class Sourcing: Fetches confirmed observations for targeted wildlife species
   via their specific taxon IDs while building an excluded, generalized 'Other'
   avian category (taxon_id=3 with target exclusions).
2. Data Scaling Tiers: Structures downloaded datasets exclusively into two volume
   tiers (T1-1000: 1000 images, T2-150: 150 images). Tier 3 (50 field/test images)
   is reserved for proprietary/manually collected test samples.
3. Automated Quality & Batch Ingestion: Handles rate limits, pagination, session
   pooling, and resolution standardization for robust data mining.
4. Programmatic & Modular Execution: Supports invocation from external pipelines
   accepting dynamic target species dictionaries, tier configurations, and output paths.
===============================================================================
"""

import os
import requests
import time
import shutil
from typing import Dict, Optional

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Default Configurations ---
HOME_DIR = os.path.expanduser("~")
DEFAULT_DATASET_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')

# Default iNaturalist Taxon IDs for Urban Wildlife
DEFAULT_TARGET_SPECIES = 
{
    "House_Sparrow": 13858,
    "Feral_Pigeon": 3017,
    "Rose_ringed_Parakeet": 18874,
    "Hooded_Crow": 133271
}

# Volume scaling tiers (T1 and T2 only; T3 is manually provided)
DEFAULT_TIERS = 
{
    "T1-1000": 1000,
    "T2-150": 150
}
BATCH_SIZE = 50

def download_verified_dataset
(
    target_species: Optional[Dict[str, int]] = None,
    tiers: Optional[Dict[str, int]] = None,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    clean_existing: bool = True
) 
-> None:
    """
    Initializes workspace and manages sequential downloads across defined target
    species and the general 'Other' class. Can be invoked directly from external scripts.
    :param target_species: Dictionary mapping species names to iNaturalist Taxon IDs.
    :param tiers: Dictionary mapping tier names to desired image counts (defaults to T1 and T2).
    :param dataset_dir: Destination root folder path for downloaded datasets.
    :param clean_existing: If True, wipes target directory before downloading.
    """
    if target_species is None:
        target_species = DEFAULT_TARGET_SPECIES
    if tiers is None:
        tiers = DEFAULT_TIERS
    print(f"[*] Initializing Dataset Acquisition Pipeline...")
    print(f"[*] Target Directory: {dataset_dir}")
    print(f"[*] Target Species ({len(target_species)}): {list(target_species.keys())}")
    print(f"[*] Active Download Tiers: {list(tiers.keys())}\n")

    if clean_existing and os.path.exists(dataset_dir):
        print("[*] Cleaning up existing dataset directory...")
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    session = requests.Session()

    # 1. Download Target Species
    for species_name, taxon_id in target_species.items():
        process_class_download(session, species_name, taxon_id, tiers, dataset_dir)
        
    # 2. Download Baseline "Other" Avian Class (Taxon ID 3: Aves, excluding target species)
    excluded_ids = ",".join(map(str, target_species.values()))
    process_class_download(session, "Other", taxon_id=3, tiers=tiers, dataset_dir=dataset_dir, exclude_ids=excluded_ids)

def process_class_download
(
    session: requests.Session,
    class_name: str,
    taxon_id: int,
    tiers: Dict[str, int],
    dataset_dir: str,
    exclude_ids: Optional[str] = None
) 
-> None:
    """
    Paginates through iNaturalist API responses and downloads verified images
    into designated tier folders for a specific species/class.
    """
    class_folder = os.path.join(dataset_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)
    print(f"\n==================================================")
    print(f"[+] Starting fetch for class: {class_name}")
    print(f"==================================================")
    page = 1
    for tier_name, limit in tiers.items():
        tier_folder = os.path.join(class_folder, tier_name)
        os.makedirs(tier_folder, exist_ok=True)
        print(f"\n[->] Filling tier {tier_name} ({limit} images)...")
        downloaded_in_tier = 0
        while downloaded_in_tier < limit:
            url = "https://api.inaturalist.org/v1/observations"
            params = 
           {
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
                    
                    # Standard medium resolution for automated dataset training
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
