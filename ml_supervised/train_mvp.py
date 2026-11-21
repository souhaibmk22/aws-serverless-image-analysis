#!/usr/bin/env python3
"""
MVP Classifier Training & Calibration Pipeline.
Uses StratifiedGroupKFold on the TRAIN set to optimize C and calibrate probabilities.
Calibrates decision thresholds (t_low for Recall >= 0.97, t_high for Precision >= 0.90)
strictly on the VALIDATION set. The Test set remains untouched.
"""
import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve, brier_score_loss

SPLITS_JSON = "data/splits.json"
FEATURES_DIR = "data/processed"
WEIGHTS_EXPORT_JSON = "ml_supervised/classifier_weights.json"


def train_and_calibrate_mvp(min_recall_low=0.97, target_precision_high=0.90):
    print("=" * 70)
    print("🚀 ENTRAÎNEMENT DU CLASSIFIEUR MVP (LOGISTIC REGRESSION + CALIBRATION)")
    print("=" * 70)

    # 1. Load splits mapping to retrieve group_id
    if not os.path.exists(SPLITS_JSON):
        print(f"❌ {SPLITS_JSON} introuvable. Exécutez data_loader.py d'abord.")
        return

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    train_records = splits.get("train", [])
    val_records = splits.get("validation", [])

    if not train_records or not val_records:
        print("❌ Splits Train ou Validation manquants.")
        return

    # Check for features
    feat_train_file = os.path.join(FEATURES_DIR, "features_train.npy")
    label_train_file = os.path.join(FEATURES_DIR, "labels_train.npy")
    feat_val_file = os.path.join(FEATURES_DIR, "features_val.npy")
    label_val_file = os.path.join(FEATURES_DIR, "labels_val.npy")

    if not (os.path.exists(feat_train_file) and os.path.exists(feat_val_file)):
        print(f"⚠️ Features non trouvées dans {FEATURES_DIR}. Lancez feature_extractor.py d'abord.")
        return

    X_train = np.load(feat_train_file)
    y_train = np.load(label_train_file)
    X_val = np.load(feat_val_file)
    y_val = np.load(label_val_file)

    groups_train = np.array([r["group_id"] for r in train_records[:len(X_train)]])

    print(f"Données chargées :")
    print(f"   • Train : {len(X_train)} échantillons ({sum(y_train==0)} Safe, {sum(y_train==1)} Dangerous)")
    print(f"   • Val   : {len(X_val)} échantillons ({sum(y_val==0)} Safe, {sum(y_val==1)} Dangerous)")

    # 2. Hyperparameter tuning on Train using StratifiedGroupKFold
    print("\n[1/3] 🔍 Optimisation de C par StratifiedGroupKFold (5 folds sur TRAIN)...")
    cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    c_candidates = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]
    best_c = 1.0
    best_brier = 1.0

    for c in c_candidates:
        brier_scores = []
        for train_idx, fold_val_idx in cv_splitter.split(X_train, y_train, groups=groups_train):
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_train[train_idx], y_train[train_idx])
            probs = clf.predict_proba(X_train[fold_val_idx])[:, 1]
            brier_scores.append(brier_score_loss(y_train[fold_val_idx], probs))
        
        mean_brier = np.mean(brier_scores)
        print(f"   • C = {c:<5} -> Brier Score OOF: {mean_brier:.4f}")
        if mean_brier < best_brier:
            best_brier = mean_brier
            best_c = c

    print(f"✅ Meilleur C sélectionné : {best_c} (Brier OOF: {best_brier:.4f})")

    # 3. Train Base Model & Calibrate with StratifiedGroupKFold
    print("\n[2/3] ⚖️ Calibration des probabilités (Platt Scaling / Sigmoid via StratifiedGroupKFold)...")
    base_clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    )
    calibrated_clf.fit(X_train, y_train, groups=groups_train)

    # 4. Threshold Calibration on the VALIDATION set only
    print("\n[3/3] 📐 Calibration des seuils ternaires sur le jeu de VALIDATION...")
    val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # Find t_low: maximum threshold that maintains Recall >= min_recall_low (0.97)
    valid_recall_idx = np.where(recalls >= min_recall_low)[0]
    if len(valid_recall_idx) > 0:
        idx_low = valid_recall_idx[-1]
        t_low = float(thresholds[min(idx_low, len(thresholds) - 1)])
    else:
        t_low = 0.20

    # Find t_high: minimum threshold that achieves Precision >= target_precision_high (0.90)
    valid_prec_idx = np.where(precisions >= target_precision_high)[0]
    if len(valid_prec_idx) > 0:
        idx_high = valid_prec_idx[0]
        t_high = float(thresholds[min(idx_high, len(thresholds) - 1)])
    else:
        t_high = 0.85

    # Ensure t_low < t_high
    if t_low >= t_high:
        t_low = max(0.05, t_high - 0.20)

    val_brier = brier_score_loss(y_val, val_probs)
    print(f"   • Seuil t_low  (Rappel cible >= {min_recall_low*100:.0f}%)    : {t_low:.4f}")
    print(f"   • Seuil t_high (Précision cible >= {target_precision_high*100:.0f}%) : {t_high:.4f}")
    print(f"   • Brier Score de validation : {val_brier:.4f}")

    # Extract weights from base model for pure NumPy export
    base_clf.fit(X_train, y_train)
    weights_data = {
        "model_type": "LogisticRegression_CLIP_ViT-B-32",
        "best_C": best_c,
        "t_low": round(t_low, 4),
        "t_high": round(t_high, 4),
        "validation_brier_score": round(float(val_brier), 4),
        "coefficients": [round(float(w), 6) for w in base_clf.coef_[0]],
        "intercept": round(float(base_clf.intercept_[0]), 6),
        "embedding_dim": len(base_clf.coef_[0])
    }

    os.makedirs(os.path.dirname(WEIGHTS_EXPORT_JSON), exist_ok=True)
    with open(WEIGHTS_EXPORT_JSON, "w") as f:
        json.dump(weights_data, f, indent=2)

    print(f"\n💾 Poids exportés en NumPy pur dans : {WEIGHTS_EXPORT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_calibrate_mvp()
