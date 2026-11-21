#!/usr/bin/env python3
"""
Sources Registry & Quota Gate Validator.
Audits image datasets, enforces the quota gate, logs counts per source & class,
and recommends legitimate manual datasets if samples are below 300.
"""
import os
import sys

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

RECOMMENDED_MANUAL_DATASETS = [
    {
        "name": "Kaggle: Weapon Detection Dataset (Guns & Knives)",
        "url": "https://www.kaggle.com/datasets/itsahmad/weapon-detection-dataset",
        "description": "Bounding box annotated images of handguns, rifles, and knives.",
        "target_class": "Handgun, Knife"
    },
    {
        "name": "Roboflow Universe: Weapon Detection",
        "url": "https://universe.roboflow.com/subhransu-sekhar/weapon-detection-o8uoi",
        "description": "Crowdsourced weapon detection dataset with pistols, combat knives, and negative backgrounds.",
        "target_class": "Pistol, Knife"
    },
    {
        "name": "Kaggle: Synthetic & Real Knife Dataset",
        "url": "https://www.kaggle.com/datasets/vencerlanz09/knife-dataset",
        "description": "Collection of combat and tactical knives across varied orientations.",
        "target_class": "Combat Knife"
    }
]


def count_images_in_dir(directory: str) -> dict:
    """Recursively counts images grouped by direct subfolder/class."""
    counts = {}
    if not os.path.exists(directory):
        return counts

    for root, _, files in os.walk(directory):
        valid_files = [f for f in files if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS]
        if valid_files:
            rel_dir = os.path.relpath(root, directory)
            counts[rel_dir] = len(valid_files)
    return counts


def audit_quota_gate(data_root="data/raw"):
    print("=" * 70)
    print("🛡️ DATASET QUOTA GATE & SOURCES AUDIT")
    print("=" * 70)

    safe_dir = os.path.join(data_root, "safe")
    dangerous_dir = os.path.join(data_root, "dangerous")
    dangerous_manual = os.path.join(data_root, "dangerous_manual")
    ood_dir = os.path.join(data_root, "ood")

    safe_counts = count_images_in_dir(safe_dir)
    dangerous_counts = count_images_in_dir(dangerous_dir)
    manual_counts = count_images_in_dir(dangerous_manual)
    ood_counts = count_images_in_dir(ood_dir)

    total_safe = sum(safe_counts.values())
    total_dangerous_auto = sum(dangerous_counts.values())
    total_dangerous_manual = sum(manual_counts.values())
    total_dangerous = total_dangerous_auto + total_dangerous_manual
    total_ood = sum(ood_counts.values())

    print(f"\n📊 1. SAFE DATASETS (Class 0): Total = {total_safe} images")
    if safe_counts:
        for sub, count in sorted(safe_counts.items()):
            print(f"   • {sub:<35} : {count:>5} images")
    else:
        print("   ⚠️  Aucune image safe ingérée pour l'instant.")

    print(f"\n🚨 2. DANGEROUS DATASETS (Class 1): Total = {total_dangerous} images")
    if dangerous_counts:
        print("   [Open Images / Sources Auto]:")
        for sub, count in sorted(dangerous_counts.items()):
            print(f"   • {sub:<35} : {count:>5} images")
    if manual_counts:
        print("   [Manual Datasets (data/raw/dangerous_manual/)]:")
        for sub, count in sorted(manual_counts.items()):
            print(f"   • {sub:<35} : {count:>5} images")
    if not dangerous_counts and not manual_counts:
        print("   ⚠️  Aucune image dangereuse détectée pour l'instant.")

    print(f"\n🌐 3. OUT-OF-DISTRIBUTION (OOD): Total = {total_ood} images")
    if ood_counts:
        for sub, count in sorted(ood_counts.items()):
            print(f"   • {sub:<35} : {count:>5} images")

    print("\n" + "-" * 70)
    print("⚖️ ÉVALUATION DE LA RÈGLE DE QUOTA:")
    print("-" * 70)

    # Hard Gate Check
    if total_dangerous < 100:
        print(f"\n❌ [STOP GATE ACTIVÉ] Nombre d'images dangereuses insuffisant : {total_dangerous} / 100 min.")
        print("   Règle stricte : L'entraînement est suspendu. Aucune donnée synthétique ne sera créée.")
        print("\n👉 Veuillez déposer au moins 100 images d'armes réelles dans :")
        print(f"   `{os.path.abspath(dangerous_manual)}`")
        print("\n💡 Datasets publics recommandés à télécharger :")
        for ds in RECOMMENDED_MANUAL_DATASETS:
            print(f"   • {ds['name']}")
            print(f"     URL: {ds['url']}")
            print(f"     Cible: {ds['target_class']}\n")
        return False

    elif total_dangerous < 300:
        print(f"\n⚠️  [AVERTISSEMENT QUOTA] Volume modéré : {total_dangerous} images dangereuses.")
        print("   Le seuil minimal d'arrêt (100) est franchi, mais un minimum de 300 images")
        print("   est requis pour des intervalles de confiance bootstrap et métriques statistiquement robustes.")
        print("\n💡 Datasets recommandés pour compléter jusqu'à 300+ :")
        for ds in RECOMMENDED_MANUAL_DATASETS:
            print(f"   • {ds['name']} -> {ds['url']}")
        return True
    else:
        print(f"\n✅ [QUOTA VALIDÉ] {total_dangerous} images dangereuses détectées (>= 300 requises).")
        print(f"   Ratio de déséquilibre (Safe/Dangerous) : {total_safe / max(1, total_dangerous):.1f}:1")
        return True


if __name__ == "__main__":
    audit_quota_gate()
