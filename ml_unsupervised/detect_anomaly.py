#!/usr/bin/env python3
"""
CLI Tool to run Unsupervised Anomaly Detection on any real image file.
Extracts features, computes distance to K-Means centroids, and classifies
whether the image is Normal or a Suspicious Anomaly.
"""
import os
import sys
from feature_extractor import extract_image_features
from kmeans_engine import KMeansAnomalyDetector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.json")

def analyze_image(image_path: str):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Le modèle n'existe pas encore. Lance d'abord : python3 ml_unsupervised/train_kmeans.py")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌ Image introuvable : {image_path}")
        sys.exit(1)

    # 1. Load trained K-Means model
    detector = KMeansAnomalyDetector()
    detector.load(MODEL_PATH)

    # 2. Read image bytes & extract 64-dim feature vector
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    features = extract_image_features(img_bytes)

    # 3. Predict Anomaly
    result = detector.predict(features)

    # 4. Display output
    print("\n" + "=" * 65)
    print(f"🔍 ANALYSE NON-SUPERVISÉE : {os.path.basename(image_path)}")
    print("=" * 65)
    print(f"  • Distance euclidienne au centroïde le plus proche : {result['min_distance_to_centroid']}")
    print(f"  • Seuil critique d'anomalie (Threshold)            : {result['threshold']}")
    print(f"  • Score d'anomalie normalisé (0.0 à 1.0)           : {result['anomaly_score']}")
    print(f"  • Cluster le plus proche                           : Cluster #{result['assigned_cluster']}")
    print("-" * 65)
    if result["is_anomaly"]:
        print(f"  RÉSULTAT : \033[1;31m{result['status']}\033[0m")
        print("  👉 L'image s'écarte trop des clusters normaux de référence !")
    else:
        print(f"  RÉSULTAT : \033[1;32m{result['status']}\033[0m")
        print("  👉 L'image correspond bien à la structure normale apprise.")
    print("=" * 65 + "\n")

    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        print("Usage: python3 ml_unsupervised/detect_anomaly.py <chemin_vers_image>")
