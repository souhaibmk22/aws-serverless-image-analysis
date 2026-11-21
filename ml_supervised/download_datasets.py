#!/usr/bin/env python3
"""
Automated Dataset Ingestion Script.
Downloads COCO val2017 into data/raw/safe/coco/ and Open Images V7 subsets
via FiftyOne by class without requiring logins.
"""
import os
import sys
import zipfile
import urllib.request

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
COCO_DEST_DIR = os.path.join(BASE_DATA_DIR, "safe", "coco")


def download_coco_val2017(dest_dir=COCO_DEST_DIR):
    """Downloads and extracts the COCO val2017 dataset (5000 images)."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "val2017.zip")

    # Check if already extracted
    existing_images = [f for f in os.listdir(dest_dir) if f.lower().endswith(".jpg")]
    if len(existing_images) >= 4900:
        print(f"✅ COCO val2017 déjà présent ({len(existing_images)} images dans {dest_dir}).")
        return len(existing_images)

    print(f"\n📦 [1/2] Téléchargement de COCO val2017 (~780 Mo)...")
    print(f"   URL: {COCO_VAL_URL}")
    print(f"   Cible: {zip_path}")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r   Téléchargement : {percent:>3}% [{mb_downloaded:>6.1f} / {total_mb:>6.1f} Mo]")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(COCO_VAL_URL, zip_path, reporthook=reporthook)
        print("\n\n📂 [2/2] Extraction des images dans data/raw/safe/coco/...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract directly flattening if inside a 'val2017/' root dir
            for member in zip_ref.infolist():
                if member.filename.lower().endswith((".jpg", ".jpeg")):
                    filename = os.path.basename(member.filename)
                    if filename:
                        target_file = os.path.join(dest_dir, filename)
                        with zip_ref.open(member) as source, open(target_file, "wb") as target:
                            target.write(source.read())

        # Cleanup zip file to preserve disk space
        if os.path.exists(zip_path):
            os.remove(zip_path)

        final_images = [f for f in os.listdir(dest_dir) if f.lower().endswith(".jpg")]
        print(f"✅ Extraction terminée : {len(final_images)} images enregistrées dans {dest_dir}.")
        return len(final_images)

    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement de COCO : {e}")
        return 0


def download_open_images_classes():
    """
    Downloads targeted classes from Open Images V7 using FiftyOne without login.
    - Dangerous classes: Handgun, Rifle, Dagger, Combat knife
    - Hard Negatives (Safe): Kitchen knife, Scissors, Axe (tool), Toy
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        print("\n🔍 Lancement de FiftyOne Zoo pour Open Images V7...")
    except ImportError:
        print("\nℹ️  FiftyOne n'est pas encore installé (pip install fiftyone).")
        print("   Vous pouvez continuer avec COCO val2017 et vos datasets manuels en attendant.")
        return

    # FiftyOne download implementation placeholder
    pass


if __name__ == "__main__":
    download_coco_val2017()
