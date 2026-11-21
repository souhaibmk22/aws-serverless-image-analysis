#!/usr/bin/env python3
"""
CLIP Feature Extractor (openai/clip-vit-base-patch32).
Generates 512-dimensional, L2-normalized embeddings for image classification.
Includes pure Pillow + NumPy preprocessing function for zero-dependency inference in Lambda.
"""
import os
import sys
import json
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import CLIPProcessor, CLIPVisionModelWithProjection
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# Standard CLIP Vision constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image_pure_pil(image_bytes: bytes) -> np.ndarray:
    """
    Pure Pillow + NumPy preprocessing matching standard HuggingFace CLIP processor.
    1. Opens image and converts to RGB
    2. Resizes shortest edge to 224 with Bicubic interpolation, then center-crops 224x224
    3. Rescales pixel values to [0.0, 1.0]
    4. Normalizes with CLIP mean and standard deviation
    5. Transposes to (1, 3, 224, 224) CHW format
    """
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Resize shortest side to 224 maintaining aspect ratio
    w, h = img.size
    if w < h:
        new_w = 224
        new_h = int(round(h * 224 / w))
    else:
        new_h = 224
        new_w = int(round(w * 224 / h))
    img = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # 2. Center crop 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    # 3. Rescale and Normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD

    # 4. Transpose from HWC to NCHW
    arr = np.transpose(arr, (2, 0, 1))  # (3, 224, 224)
    return np.expand_dims(arr, axis=0)   # (1, 3, 224, 224)


class CLIPExtractor:
    def __init__(self, model_name=CLIP_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on device: {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device).eval()

    def extract_single_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            inputs = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.image_embeds.cpu().numpy()[0]
                # L2-normalization
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding

    def extract_batch(self, image_paths: list, batch_size=32) -> np.ndarray:
        all_embeddings = []
        valid_indices = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_valid = []

            for idx, p in enumerate(batch_paths):
                try:
                    with Image.open(p) as img:
                        batch_images.append(img.convert("RGB"))
                        batch_valid.append(i + idx)
                except Exception as e:
                    print(f"⚠️ Skipped corrupted image: {p} ({e})")

            if not batch_images:
                continue

            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeds = outputs.image_embeds.cpu().numpy()
                # L2 normalize each vector
                norms = np.linalg.norm(embeds, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeds = embeds / norms
                all_embeddings.append(embeds)
                valid_indices.extend(batch_valid)

        if not all_embeddings:
            return np.empty((0, EMBEDDING_DIM)), []
        return np.vstack(all_embeddings), valid_indices


def extract_features_for_splits(splits_json="data/splits.json", output_dir="data/processed"):
    if not HAS_TRANSFORMERS:
        print("❌ transformers / torch non installés. Veuillez installer requirements.txt d'abord.")
        return

    if not os.path.exists(splits_json):
        print(f"❌ Fichier de splits introuvable : {splits_json}")
        return

    with open(splits_json) as f:
        splits = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    extractor = CLIPExtractor()

    for s_name in ["train", "validation", "test", "ood"]:
        records = splits.get(s_name, [])
        if not records:
            continue

        print(f"\n🔄 Extraction des embeddings CLIP pour le split '{s_name}' ({len(records)} images)...")
        paths = [r["path"] for r in records]
        raw_labels = np.array([r["label"] for r in records], dtype=np.int32)

        embeddings, valid_idx = extractor.extract_batch(paths)
        filtered_labels = raw_labels[valid_idx]

        feat_path = os.path.join(output_dir, f"features_{s_name}.npy")
        label_path = os.path.join(output_dir, f"labels_{s_name}.npy")

        np.save(feat_path, embeddings.astype(np.float32))
        np.save(label_path, filtered_labels)

        print(f"✅ Sauvegardé : {feat_path} ({embeddings.shape}), {label_path}")


if __name__ == "__main__":
    extract_features_for_splits()
