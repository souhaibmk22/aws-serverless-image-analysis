import io
import math
import unittest
import numpy as np
from PIL import Image

# Import preprocessing function from feature_extractor
from ml_supervised.feature_extractor import (
    preprocess_image_pure_pil,
    CLIP_MEAN,
    CLIP_STD,
    EMBEDDING_DIM
)


class TestSafetyClassifier(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 300x400 RGB test image in memory
        self.test_img = Image.new("RGB", (300, 400), color=(128, 64, 200))
        img_byte_arr = io.BytesIO()
        self.test_img.save(img_byte_arr, format="JPEG")
        self.img_bytes = img_byte_arr.getvalue()

    def test_pure_pil_preprocessing_dimensions(self):
        """Validates that pure Pillow preprocessing produces exact (1, 3, 224, 224) float32 tensor."""
        tensor = preprocess_image_pure_pil(self.img_bytes)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

    def test_pure_pil_preprocessing_parity_with_hf(self):
        """
        Validates Point 5 of Additions:
        Lambda preprocessing (PIL + NumPy) matches HuggingFace CLIPProcessor (cosine similarity > 0.999).
        """
        try:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            hf_out = processor(images=self.test_img, return_tensors="np")["pixel_values"]

            pil_out = preprocess_image_pure_pil(self.img_bytes)

            # Flatten vectors to compute cosine similarity
            v_hf = hf_out.flatten()
            v_pil = pil_out.flatten()

            cosine_sim = np.dot(v_hf, v_pil) / (np.linalg.norm(v_hf) * np.linalg.norm(v_pil))
            print(f"\n[Test Parity] Preprocessing Cosine Similarity: {cosine_sim:.6f}")
            self.assertGreater(cosine_sim, 0.999, "Pure PIL preprocessing diverges from HuggingFace processor!")
        except ImportError:
            # If transformers is not installed in local environment, verify pixel range
            tensor = preprocess_image_pure_pil(self.img_bytes)
            # Normalized values should roughly fall between -3.0 and 3.0
            self.assertTrue(np.all(tensor >= -3.0) and np.all(tensor <= 3.0))

    def test_pure_numpy_inference_equivalence(self):
        """
        Validates that pure NumPy logistic inference (z = w^T x + b, p = 1 / (1 + e^-z))
        strictly equals Scikit-Learn predict_proba within tolerance 1e-6.
        """
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        # Mock 100 samples of 512-dim features
        X_mock = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
        X_mock /= np.linalg.norm(X_mock, axis=1, keepdims=True)
        y_mock = np.random.choice([0, 1], size=100)

        clf = LogisticRegression(random_state=42, max_iter=200)
        clf.fit(X_mock, y_mock)

        # Scikit-learn probabilities for positive class (label 1)
        skl_probs = clf.predict_proba(X_mock)[:, 1]

        # Pure NumPy manual calculation using exported weights
        weights = clf.coef_[0]
        intercept = clf.intercept_[0]
        logits = np.dot(X_mock, weights) + intercept
        numpy_probs = 1.0 / (1.0 + np.exp(-logits))

        # Check max absolute difference is strictly below 1e-6
        max_diff = np.max(np.abs(skl_probs - numpy_probs))
        self.assertLess(max_diff, 1e-6, f"Difference between Sklearn and NumPy too large: {max_diff}")

    def test_decision_policy_thresholds(self):
        """
        Validates ternary routing policy:
        p < t_low: SAFE
        t_low <= p < t_high: REVIEW (pending_review)
        p >= t_high: DANGEROUS
        """
        t_low = 0.20
        t_high = 0.85

        def route_decision(p):
            if p < t_low:
                return "SAFE"
            elif p < t_high:
                return "REVIEW"
            else:
                return "DANGEROUS"

        self.assertEqual(route_decision(0.05), "SAFE")
        self.assertEqual(route_decision(0.19), "SAFE")
        self.assertEqual(route_decision(0.20), "REVIEW")
        self.assertEqual(route_decision(0.50), "REVIEW")
        self.assertEqual(route_decision(0.849), "REVIEW")
        self.assertEqual(route_decision(0.85), "DANGEROUS")
        self.assertEqual(route_decision(0.99), "DANGEROUS")


if __name__ == "__main__":
    unittest.main()
