#!/usr/bin/env python3
"""
Visual Inspection Grid Generator (Contact Sheet).
Generates an HTML contact sheet with 50 random samples per class per source
to allow human audit of labels before feature extraction and training.
"""
import os
import random
import base64
from sources_registry import count_images_in_dir, VALID_EXTENSIONS

GRID_OUTPUT_HTML = "data/visual_audit/inspection_grid.html"


def generate_inspection_grid(data_root="data/raw", samples_per_class=50):
    print("=" * 70)
    print("👁️ GÉNÉRATION DE LA PLANCHE D'AUDIT VISUEL (50 IMAGES / SOURCE)")
    print("=" * 70)

    categories = {
        "SAFE (COCO / Extra)": os.path.join(data_root, "safe"),
        "DANGEROUS (Armes réelles)": os.path.join(data_root, "dangerous"),
        "DANGEROUS (Manuel / Kaggle / Roboflow)": os.path.join(data_root, "dangerous_manual"),
        "OUT-OF-DISTRIBUTION (OOD)": os.path.join(data_root, "ood"),
    }

    html_sections = []
    total_sampled = 0
    random.seed(42)

    for cat_name, cat_dir in categories.items():
        if not os.path.exists(cat_dir):
            continue

        # Find all valid image paths
        all_imgs = []
        for root, _, files in os.walk(cat_dir):
            for f in files:
                if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS:
                    all_imgs.append(os.path.join(root, f))

        if not all_imgs:
            continue

        selected = random.sample(all_imgs, min(samples_per_class, len(all_imgs)))
        total_sampled += len(selected)

        cards = []
        for img_path in selected:
            rel_path = os.path.relpath(img_path, os.path.dirname(GRID_OUTPUT_HTML))
            filename = os.path.basename(img_path)
            subfolder = os.path.basename(os.path.dirname(img_path))
            cards.append(f"""
                <div class="border border-slate-700 bg-slate-900 rounded-lg p-2 flex flex-col justify-between">
                    <div class="h-32 w-full bg-slate-950 rounded flex items-center justify-center overflow-hidden mb-2">
                        <img src="{rel_path}" loading="lazy" class="max-h-32 object-contain" alt="{filename}" onerror="this.parentElement.innerHTML='<span class=text-xs>Aperçu indisponible</span>'">
                    </div>
                    <div class="text-[11px] text-slate-300 font-mono truncate" title="{filename}">{filename}</div>
                    <div class="text-[10px] text-blue-400 font-semibold">{subfolder}</div>
                </div>
            """)

        html_sections.append(f"""
            <div class="mb-8">
                <div class="flex items-center justify-between border-b border-slate-700 pb-2 mb-4">
                    <h2 class="text-base font-bold text-white">{cat_name}</h2>
                    <span class="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">{len(selected)} / {len(all_imgs)} images</span>
                </div>
                <div class="grid grid-cols-2 sm:grid-cols-5 md:grid-cols-10 gap-2">
                    {''.join(cards)}
                </div>
            </div>
        """)

    if total_sampled == 0:
        print("⚠️  Aucune image trouvée à afficher dans data/raw/.")
        return False

    os.makedirs(os.path.dirname(GRID_OUTPUT_HTML), exist_ok=True)
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Planche d'Audit Visuel - 50 Images / Classe</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 p-6 font-sans">
    <div class="max-w-7xl mx-auto">
        <header class="mb-8 border-b border-slate-800 pb-4">
            <h1 class="text-xl font-bold text-white">🔍 Planche de Contrôle & Validation Visuelle des Labels</h1>
            <p class="text-xs text-slate-400 mt-1">
                Échantillonnage aléatoire de 50 images par classe et par source. Veuillez inspecter les vignettes ci-dessous pour confirmer qu'aucun couteau de cuisine ou outil n'est étiqueté DANGEROUS.
            </p>
        </header>
        {''.join(html_sections)}
    </div>
</body>
</html>"""

    with open(GRID_OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Planche de contact générée avec succès : {GRID_OUTPUT_HTML}")
    print(f"   Total d'échantillons audités : {total_sampled} images.")
    print("👉 Ouvrez ce fichier dans votre navigateur pour valider les labels avant l'entraînement.")
    return True


if __name__ == "__main__":
    generate_inspection_grid()
