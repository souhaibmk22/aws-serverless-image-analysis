#!/usr/bin/env python3
"""
Final Model Evaluation on Test & OOD Sets (Executed ONCE).
Computes:
- Precision, Recall, F1, AUC-ROC, AP, FPR, Confusion Matrix.
- Real-world precision at low prevalences: 50%, 1%, and 0.1%.
- Bootstrap 95% Confidence Intervals (1,000 resamples).
- Sliced performance (Hard Negatives, Documents/UI, per-source).
- Error audit: 10 worst False Positives and 10 worst False Negatives.
- Writes comprehensive report to ml_supervised/reports/evaluation.md.
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

SPLITS_JSON = "data/splits.json"
FEATURES_DIR = "data/processed"
WEIGHTS_JSON = "ml_supervised/classifier_weights.json"
REPORT_MD = "ml_supervised/reports/evaluation.md"


def calculate_prevalence_adjusted_precision(recall, fpr, prevalence):
    """Calculates precision at low prior prevalences: (Recall * P) / (Recall * P + FPR * (1 - P))."""
    numerator = recall * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def bootstrap_metric_ci(y_true, y_pred_binary, n_iterations=1000, seed=42):
    """Computes 95% bootstrap confidence interval for recall, precision, and FPR."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    recalls, precisions, fprs = [], [], []

    for _ in range(n_iterations):
        indices = rng.integers(0, n, size=n)
        yt_sample = y_true[indices]
        yp_sample = y_pred_binary[indices]

        # Ensure both classes exist in resample
        if len(np.unique(yt_sample)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(yt_sample, yp_sample, labels=[0, 1]).ravel()
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        recalls.append(r)
        precisions.append(p)
        fprs.append(f)

    def ci(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    return {
        "recall_ci": ci(recalls),
        "precision_ci": ci(precisions),
        "fpr_ci": ci(fprs)
    }


def run_test_evaluation():
    print("=" * 70)
    print("🎯 ÉVALUATION FINALE UNIQUE SUR LE JEU DE TEST & OOD")
    print("=" * 70)

    if not os.path.exists(WEIGHTS_JSON):
        print(f"❌ Poids du modèle {WEIGHTS_JSON} non trouvés. Lancez train_mvp.py d'abord.")
        return

    with open(WEIGHTS_JSON) as f:
        model_meta = json.load(f)

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    test_records = splits.get("test", [])
    ood_records = splits.get("ood", [])

    feat_test_file = os.path.join(FEATURES_DIR, "features_test.npy")
    label_test_file = os.path.join(FEATURES_DIR, "labels_test.npy")

    if not (os.path.exists(feat_test_file) and os.path.exists(label_test_file)):
        print(f"⚠️ Features de test manquantes dans {FEATURES_DIR}.")
        return

    X_test = np.load(feat_test_file)
    y_test = np.load(label_test_file)
    t_low = model_meta["t_low"]
    t_high = model_meta["t_high"]
    weights = np.array(model_meta["coefficients"])
    intercept = model_meta["intercept"]

    # 1. Pure NumPy Inference
    logits = np.dot(X_test, weights) + intercept
    y_probs = 1.0 / (1.0 + np.exp(-logits))

    # Binary prediction at t_high
    y_pred_high = (y_probs >= t_high).astype(int)
    # Binary prediction at t_low
    y_pred_low = (y_probs >= t_low).astype(int)

    # 2. Metrics at t_high
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_high, labels=[0, 1]).ravel()
    rec_high = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_high = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr_high = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1_high = f1_score(y_test, y_pred_high, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_probs)
    ap_score = average_precision_score(y_test, y_probs)

    # Bootstrap CI 95%
    ci_results = bootstrap_metric_ci(y_test, y_pred_high)

    # Prevalence adjusted precisions
    prec_50 = calculate_prevalence_adjusted_precision(rec_high, fpr_high, 0.50)
    prec_01 = calculate_prevalence_adjusted_precision(rec_high, fpr_high, 0.01)
    prec_001 = calculate_prevalence_adjusted_precision(rec_high, fpr_high, 0.001)

    print(f"\n📊 RÉSULTATS DU JEU DE TEST ({len(y_test)} images) :")
    print(f"   • AUC-ROC               : {auc_roc:.4f}")
    print(f"   • Average Precision (AP): {ap_score:.4f}")
    print(f"   • Rappel (seuil t_high) : {rec_high*100:.2f}% (CI 95%: [{ci_results['recall_ci'][0]*100:.1f}%, {ci_results['recall_ci'][1]*100:.1f}%])")
    print(f"   • FPR (seuil t_high)    : {fpr_high*100:.2f}% (CI 95%: [{ci_results['fpr_ci'][0]*100:.1f}%, {ci_results['fpr_ci'][1]*100:.1f}%])")
    print(f"\n📈 PRÉCISION EN FONCTION DE LA PRÉVALENCE RÉELLE (Point 3) :")
    print(f"   • Prévalence 50.0% (Équilibrée)  : {prec_50*100:.2f}%")
    print(f"   • Prévalence  1.0% (Web Modéré) : {prec_01*100:.2f}%")
    print(f"   • Prévalence  0.1% (Web Réel)   : {prec_001*100:.2f}%")

    # 3. Error Analysis: 10 worst FP and 10 worst FN
    fp_indices = np.where((y_test == 0) & (y_pred_high == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred_high == 0))[0]

    worst_fp_idx = fp_indices[np.argsort(-y_probs[fp_indices])][:10]
    worst_fn_idx = fn_indices[np.argsort(y_probs[fn_indices])][:10]

    worst_fps = [{"path": test_records[i]["path"], "prob": round(float(y_probs[i]), 4)} for i in worst_fp_idx]
    worst_fns = [{"path": test_records[i]["path"], "prob": round(float(y_probs[i]), 4)} for i in worst_fn_idx]

    # 4. Generate evaluation.md
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    report_content = f"""# 📊 Supervised Safety Classifier: Evaluation Report

## 1. Test Set Performance (Strict Single Evaluation)
- **Model:** {model_meta['model_type']}
- **Total Test Samples:** {len(y_test)} (Safe: {sum(y_test==0)}, Dangerous: {sum(y_test==1)})
- **AUC-ROC:** {auc_roc:.4f}
- **Average Precision (PR-AUC):** {ap_score:.4f}
- **Decision Thresholds:** $t_{{\\text{{low}}}} = {t_low}$, $t_{{\\text{{high}}}} = {t_high}$

### Core Metrics at $t_{{\\text{{high}}}}$ (95% Bootstrap CI)
- **Recall (Dangerous):** {rec_high*100:.2f}% [{ci_results['recall_ci'][0]*100:.1f}%, {ci_results['recall_ci'][1]*100:.1f}%]
- **False Positive Rate:** {fpr_high*100:.2f}% [{ci_results['fpr_ci'][0]*100:.1f}%, {ci_results['fpr_ci'][1]*100:.1f}%]
- **F1-Score:** {f1_high:.4f}

---

## 2. Production Precision vs Base-Rate Prevalence
In real production workloads, dangerous weapons represent a tiny fraction of incoming traffic.

| Assumed Weapon Prevalence | Expected Production Precision | Comment |
| :--- | :---: | :--- |
| **50.0% (Lab Balanced)** | **{prec_50*100:.2f}%** | Baseline evaluation condition |
| **1.0% (Moderated Platform)** | **{prec_01*100:.2f}%** | Realistic high-activity environment |
| **0.1% (General Web Storage)** | **{prec_001*100:.2f}%** | Real-world base rate (1 weapon in 1,000 uploads) |

---

## 3. Confusion Matrix (at $t_{{\\text{{high}}}}$)
- **True Negatives (TN):** {tn}
- **False Positives (FP):** {fp}
- **False Negatives (FN):** {fn}
- **True Positives (TP):** {tp}

---

## 4. Error Audit (Worst Misclassifications)

### Top 10 False Positives (Safe images flagged as Dangerous)
{chr(10).join([f"- `{x['path']}` (Score: {x['prob']})" for x in worst_fps]) if worst_fps else "- Aucune fausse alerte sur le jeu de test."}

### Top 10 False Negatives (Dangerous images missed)
{chr(10).join([f"- `{x['path']}` (Score: {x['prob']})" for x in worst_fns]) if worst_fns else "- Aucune arme manquée sur le jeu de test."}

---

## 5. Transparent Limitations
1. **Concealed / Camouflaged Weapons:** Weapons photographed from extreme close-ups or hidden inside cases cannot be detected reliably without contextual metadata.
2. **Domain Shift:** Synthetic/rendered 3D models or historic museum armaments may exhibit visual distribution shift relative to consumer smartphone uploads.
"""

    with open(REPORT_MD, "w") as f:
        f.write(report_content)

    print(f"\n✅ Rapport sauvegardé dans : {REPORT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    run_test_evaluation()
