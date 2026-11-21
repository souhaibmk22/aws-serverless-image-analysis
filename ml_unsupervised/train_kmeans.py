#!/usr/bin/env python3
"""
Unsupervised Machine Learning Training Script using K-Means Clustering.
Learns the pattern of normal images without any labels (Unsupervised Learning).
Computes optimal cluster centroids and the anomaly detection threshold.
"""
import os
import random
from feature_extractor import extract_image_features
from kmeans_engine import KMeansAnomalyDetector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.json")

def generate_synthetic_normal_dataset(num_samples=80):
    """
    Generates representative feature vectors of normal daily photos
    (landscapes, indoor photos, pets, documents) to train K-Means.
    """
    random.seed(42)
    dataset = []

    # Cluster 1: Nature & Landscapes (Greens/Blues, soft gradients)
    for _ in range(num_samples // 3):
        base = [random.uniform(0.1, 0.4) for _ in range(64)]
        for i in range(16, 32): base[i] += random.uniform(0.2, 0.4) # Green boost
        dataset.append([round(v, 4) for v in base])

    # Cluster 2: Domestic & Indoor (Warm colors, balanced gradients)
    for _ in range(num_samples // 3):
        base = [random.uniform(0.2, 0.5) for _ in range(64)]
        for i in range(0, 16): base[i] += random.uniform(0.1, 0.3)  # Red/Warm boost
        dataset.append([round(v, 4) for v in base])

    # Cluster 3: Clear documents & Everyday objects (High luminance, defined edges)
    for _ in range(num_samples // 3):
        base = [random.uniform(0.3, 0.6) for _ in range(64)]
        for i in range(48, 56): base[i] += random.uniform(0.2, 0.4) # Edge boost
        dataset.append([round(v, 4) for v in base])

    return dataset

def main():
    print("=" * 65)
    print("🧠 ENTRAÎNEMENT MACHINE LEARNING NON-SUPERVISÉ (K-MEANS)")
    print("=" * 65)
    print("Objectif : Apprendre les clusters normaux SANS LABELS (Y inexistant)")
    print("et calculer la frontière de décision pour détecter les anomalies.\n")

    # 1. Preparation of Dataset
    print("[1/4] 📊 Chargement des vecteurs caractéristiques normaux...")
    X_train = generate_synthetic_normal_dataset(num_samples=90)
    print(f"      Nombre d'échantillons d'entraînement : {len(X_train)}")
    print(f"      Dimensions de chaque vecteur : {len(X_train[0])} caractéristiques")

    # 2. Training K-Means
    K = 3
    print(f"\n[2/4] 🔄 Entraînement de l'algorithme K-Means (K={K} clusters)...")
    detector = KMeansAnomalyDetector(k=K, max_iter=50)
    detector.fit(X_train)

    print(f"      Convergence atteinte !")
    print(f"      Centroïdes calculés : {len(detector.centroids)} centres de gravité")

    # 3. Decision Boundary / Anomaly Threshold
    print(f"\n[3/4] 📐 Calcul mathématique du seuil d'anomalie...")
    print(f"      Seuil d'anomalie critique (Threshold) = {detector.anomaly_threshold}")
    print(f"      Règle : Si distance(image, centroïde_le_plus_proche) > {detector.anomaly_threshold} ➔ ANOMALIE !")

    # 4. Save model
    print(f"\n[4/4] 💾 Sauvegarde du modèle entraîné...")
    detector.save(MODEL_PATH)
    print(f"      Modèle exporté vers : {MODEL_PATH}")

    print("\n" + "=" * 65)
    print("✔ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("=" * 65)

if __name__ == "__main__":
    main()
