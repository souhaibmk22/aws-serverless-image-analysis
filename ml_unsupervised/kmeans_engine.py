import json
import math
import random

def euclidean_distance(v1: list, v2: list) -> float:
    """Computes Euclidean distance between two feature vectors: sqrt(sum((a_i - b_i)^2))."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

class KMeansAnomalyDetector:
    """
    Unsupervised K-Means Clustering and Anomaly Detection Engine.
    Clusters normal images and identifies outliers by computing the distance to the nearest centroid.
    """
    def __init__(self, k: int = 3, max_iter: int = 50):
        self.k = k
        self.max_iter = max_iter
        self.centroids = []
        self.anomaly_threshold = 0.0

    def fit(self, X: list):
        """
        Trains K-Means on normal images (X: list of feature vectors).
        Calculates cluster centroids and the decision boundary threshold.
        """
        if len(X) < self.k:
            raise ValueError(f"Need at least {self.k} samples to train {self.k} clusters.")

        # 1. K-Means Initialization (Random selection)
        random.seed(42)
        self.centroids = [list(c) for c in random.sample(X, self.k)]

        for iteration in range(self.max_iter):
            # Cluster assignment step
            clusters = [[] for _ in range(self.k)]
            for vec in X:
                distances = [euclidean_distance(vec, c) for c in self.centroids]
                nearest_cluster = distances.index(min(distances))
                clusters[nearest_cluster].append(vec)

            # Centroid update step
            new_centroids = []
            for cluster_idx, members in enumerate(clusters):
                if members:
                    dim = len(members[0])
                    new_center = [
                        sum(m[d] for m in members) / len(members) for d in range(dim)
                    ]
                    new_centroids.append(new_center)
                else:
                    new_centroids.append(self.centroids[cluster_idx])

            # Check convergence
            shifts = sum(
                euclidean_distance(c_old, c_new)
                for c_old, c_new in zip(self.centroids, new_centroids)
            )
            self.centroids = new_centroids
            if shifts < 1e-4:
                break

        # 2. Compute Anomaly Decision Boundary (Threshold)
        distances_to_centers = []
        for vec in X:
            min_dist = min(euclidean_distance(vec, c) for c in self.centroids)
            distances_to_centers.append(min_dist)

        mean_d = sum(distances_to_centers) / len(distances_to_centers)
        variance_d = sum((d - mean_d) ** 2 for d in distances_to_centers) / len(distances_to_centers)
        std_d = math.sqrt(variance_d)

        # Statistical Threshold: Mean + 2.5 * Standard Deviation (99% confidence interval)
        self.anomaly_threshold = round(mean_d + (2.5 * std_d), 4)

    def predict(self, feature_vector: list) -> dict:
        """
        Predicts if a new image vector is an Anomaly (Suspicious) or Normal.
        Returns distance, threshold, anomaly status, and confidence score.
        """
        distances = [euclidean_distance(feature_vector, c) for c in self.centroids]
        min_distance = min(distances)
        nearest_cluster = distances.index(min_distance)

        is_anomaly = min_distance > self.anomaly_threshold
        # Compute normalized anomaly score (0.0 to 1.0)
        anomaly_score = round(min(1.0, min_distance / (self.anomaly_threshold * 1.5)), 3)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "min_distance_to_centroid": round(min_distance, 4),
            "threshold": self.anomaly_threshold,
            "assigned_cluster": nearest_cluster,
            "status": "🚨 SUSPICIOUS / ANOMALY" if is_anomaly else "✅ NORMAL / CONFORME"
        }

    def save(self, filepath: str):
        data = {
            "k": self.k,
            "centroids": self.centroids,
            "anomaly_threshold": self.anomaly_threshold
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.k = data["k"]
        self.centroids = data["centroids"]
        self.anomaly_threshold = data["anomaly_threshold"]
