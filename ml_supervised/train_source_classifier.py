#!/usr/bin/env python3
"""
Source Domain Classifier (Anti-Shortcut Learning Audit).
Trains a multi-class classifier on CLIP embeddings to predict the image source dataset.
If accuracy is significantly above random chance (1 / N_sources), raises a warning
that the downstream safety classifier risks learning dataset visual styles rather than true semantics.
"""
import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

SPLITS_JSON = "data/splits.json"
FEATURES_DIR = "data/processed"


def audit_source_shortcuts(splits_json=SPLITS_JSON, features_dir=FEATURES_DIR):
    print("=" * 70)
    print("🔍 AUDIT ANTI-RACCOURCI : SOURCE DOMAIN CLASSIFIER")
    print("=" * 70)

    if not os.path.exists(splits_json):
        print(f"❌ Splits introuvables : {splits_json}. Exécutez data_loader.py d'abord.")
        return

    with open(splits_json) as f:
        splits = json.load(f)

    # Use training split for source classification
    train_records = splits.get("train", [])
    if not train_records:
        print("❌ Split d'entraînement vide.")
        return

    sources = [r["source"] for r in train_records]
    unique_sources = sorted(list(set(sources)))
    n_sources = len(unique_sources)

    print(f"Sources détectées ({n_sources}) :")
    source_counts = {s: sources.count(s) for s in unique_sources}
    for s, count in source_counts.items():
        print(f"   • {s:<30} : {count:>5} images")

    if n_sources <= 1:
        print("ℹ️ Une seule source détectée dans le train. Audit de biais inter-sources non applicable.")
        return

    # Load pre-extracted training features if available
    feat_train_file = os.path.join(features_dir, "features_train.npy")
    if not os.path.exists(feat_train_file):
        print(f"⚠️ Features {feat_train_file} non trouvées. Veuillez extraire les features d'abord.")
        return

    X = np.load(feat_train_file)
    source_to_idx = {s: i for i, s in enumerate(unique_sources)}
    y = np.array([source_to_idx[s] for s in sources[:len(X)]])

    # Theoretical random chance baseline
    majority_class_pct = max(source_counts.values()) / len(sources)
    random_chance_pct = 1.0 / n_sources

    print(f"\nBaseline de référence :")
    print(f"   • Hasard uniforme pur (1/N)       : {random_chance_pct * 100:.1f}%")
    print(f"   • Classe majoritaire naïve        : {majority_class_pct * 100:.1f}%")

    print("\n🔄 Entraînement du classifieur de domaine (5-Fold Stratified CV)...")
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    print(f"✅ Précision mesurée du Source Classifier : {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")

    # Audit Threshold & Warning
    # If accuracy exceeds majority class baseline by > 15 percentage points
    if mean_acc > (majority_class_pct + 0.15):
        print("\n" + "⚠️ " * 20)
        print("🚨 ALERTE BIAIS DE DOMAINE DÉTECTÉE (SHORTCUT LEARNING RISK) !")
        print(f"   Le modèle prédit la source à {mean_acc*100:.1f}% (bien au-dessus de {majority_class_pct*100:.1f}%).")
        print("   Explication : Les embeddings CLIP capturent des particularités stylistiques")
        print("   propres aux datasets (ex: compression, ratio, bokeh) plutôt que la sémantique pure.")
        print("   Recommandation : S'assurer que les Hard Negatives et les Dangerous partagent")
        print("   des sources communes (Open Images) pour neutraliser le raccourci.")
        print("⚠️ " * 20 + "\n")
        return False
    else:
        print("\n✅ [AUDIT VALIDÉ] Biais de domaine maîtrisé. L'espace d'embedding ne permet pas de sur-séparer trivialement les sources.")
        return True


if __name__ == "__main__":
    audit_source_shortcuts()
