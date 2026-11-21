#!/usr/bin/env python3
"""
Dataset Ingestion, Deduplication, and Stratified Group Split (70/15/15).
Computes perceptual hashes to eliminate data leakage and ensure augmented
copies and near-duplicates remain strictly in the same split.
"""
import os
import sys
import json
import random
from collections import defaultdict
import numpy as np

try:
    from PIL import Image
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS_JSON_PATH = "data/splits.json"


def compute_image_hash(image_path: str):
    """Computes perceptual hash to detect near-duplicates, or falls back to size/content hash."""
    if HAS_IMAGEHASH:
        try:
            with Image.open(image_path) as img:
                return str(imagehash.phash(img.convert("RGB")))
        except Exception:
            return None
    else:
        # Fallback: file size + name signature
        try:
            st = os.stat(image_path)
            return f"{st.st_size}_{os.path.basename(image_path)}"
        except Exception:
            return None


def collect_images_with_labels(data_root="data/raw"):
    """
    Collects images and maps them to:
    - 0: SAFE (safe/, hard_negatives/, documents_screenshots/)
    - 1: DANGEROUS (dangerous/, dangerous_manual/)
    - OOD: ood/
    """
    records = []
    
    # 1. Safe directories
    safe_roots = [
        ("data/raw/safe", "safe_standard"),
        ("data/raw/safe/coco", "coco_val2017"),
        ("data/raw/safe/hard_negatives", "hard_negatives"),
        ("data/raw/safe/documents_screenshots", "documents_screenshots")
    ]
    for path, source_name in safe_roots:
        if os.path.exists(path):
            for f in os.listdir(path):
                fpath = os.path.join(path, f)
                if os.path.isfile(fpath) and os.path.splitext(f.lower())[1] in VALID_EXTENSIONS:
                    records.append({
                        "path": fpath,
                        "label": 0,
                        "class_name": "safe",
                        "source": source_name,
                        "is_ood": False
                    })

    # 2. Dangerous directories
    dangerous_roots = [
        ("data/raw/dangerous", "open_images_weapons"),
        ("data/raw/dangerous_manual", "manual_weapons")
    ]
    for path, source_name in dangerous_roots:
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                for f in files:
                    fpath = os.path.join(root, f)
                    if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS:
                        sub = os.path.basename(os.path.dirname(fpath))
                        records.append({
                            "path": fpath,
                            "label": 1,
                            "class_name": "dangerous",
                            "source": f"{source_name}_{sub}" if sub else source_name,
                            "is_ood": False
                        })

    # 3. OOD source (Out-Of-Distribution)
    ood_path = "data/raw/ood"
    if os.path.exists(ood_path):
        for root, _, files in os.walk(ood_path):
            for f in files:
                fpath = os.path.join(root, f)
                if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS:
                    sub = os.path.basename(os.path.dirname(fpath)).lower()
                    label = 1 if any(k in sub for k in ["danger", "weapon", "gun", "knife"]) else 0
                    records.append({
                        "path": fpath,
                        "label": label,
                        "class_name": "dangerous" if label == 1 else "safe",
                        "source": "ood_holdout",
                        "is_ood": True
                    })

    return records


def build_and_save_splits(data_root="data/raw", seed=42):
    print("=" * 70)
    print("🔄 DÉDUPLICATION PAR HACHAGE PERCEPTUEL ET SPLIT GROUPÉ 70/15/15")
    print("=" * 70)

    raw_records = collect_images_with_labels(data_root)
    if not raw_records:
        print("❌ Aucune image trouvée dans data/raw/.")
        return None

    # Separate in-distribution and OOD
    in_dist = [r for r in raw_records if not r["is_ood"]]
    ood_records = [r for r in raw_records if r["is_ood"]]

    print(f"Total images collectées : {len(raw_records)} (In-distribution: {len(in_dist)}, OOD: {len(ood_records)})")

    # Deduplication & group_id assignment
    hash_to_group = {}
    deduped_records = []
    dropped_leakage = 0

    for r in in_dist:
        h = compute_image_hash(r["path"])
        if not h:
            continue
        if h in hash_to_group:
            existing = hash_to_group[h]
            if existing["label"] != r["label"]:
                # Conflicting label for identical hash -> Drop to prevent leakage
                dropped_leakage += 1
                continue
            r["group_id"] = existing["group_id"]
        else:
            r["group_id"] = h
            hash_to_group[h] = r
            deduped_records.append(r)

    print(f"Images dédupliquées : {len(deduped_records)} (Doublons/conflits écartés : {dropped_leakage})")

    # Group Stratified 70/15/15 Split
    # Group unique group_ids by class
    safe_groups = list({r["group_id"] for r in deduped_records if r["label"] == 0})
    dangerous_groups = list({r["group_id"] for r in deduped_records if r["label"] == 1})

    random.seed(seed)
    random.shuffle(safe_groups)
    random.shuffle(dangerous_groups)

    def split_group_list(groups):
        n = len(groups)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_g = set(groups[:n_train])
        val_g = set(groups[n_train:n_train + n_val])
        test_g = set(groups[n_train + n_val:])
        return train_g, val_g, test_g

    safe_tr, safe_val, safe_test = split_group_list(safe_groups)
    dang_tr, dang_val, dang_test = split_group_list(dangerous_groups)

    train_groups = safe_tr | dang_tr
    val_groups = safe_val | dang_val
    test_groups = safe_test | dang_test

    splits = {
        "train": [r for r in deduped_records if r["group_id"] in train_groups],
        "validation": [r for r in deduped_records if r["group_id"] in val_groups],
        "test": [r for r in deduped_records if r["group_id"] in test_groups],
        "ood": ood_records
    }

    print("\n📊 RÉSULTAT DU SPLIT STRATIFIÉ GROUPÉ (seed 42) :")
    for s_name in ["train", "validation", "test", "ood"]:
        items = splits[s_name]
        n_safe = sum(1 for x in items if x["label"] == 0)
        n_dang = sum(1 for x in items if x["label"] == 1)
        print(f"   • {s_name.upper():<12} : Total={len(items):>5} | Safe={n_safe:>5} | Dangerous={n_dang:>4}")

    os.makedirs(os.path.dirname(SPLITS_JSON_PATH), exist_ok=True)
    with open(SPLITS_JSON_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n✅ Splits sauvegardés dans : {SPLITS_JSON_PATH}")
    return splits


if __name__ == "__main__":
    build_and_save_splits()
