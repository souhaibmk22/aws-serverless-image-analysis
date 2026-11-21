# 📜 DATA SOURCES & LICENSE REGISTRY

This registry audits all data sources utilized in the Supervised Image Safety Classifier project.
Every source is cataloged with its canonical URL, licensing terms, image counts, and critical license caveats.

---

## 🛡️ License Terms & Attribution

| Dataset Source | Canonical URL | Intended Class | License Type | Commercial Use | License Caveat & Attribution |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **COCO val2017** | [cocodataset.org](https://cocodataset.org/) | Safe | Varied Flickr Licenses | Per-Image | **Per-Image Caveat:** COCO images are sourced from Flickr. Each image carries an individual Flickr license (CC-BY 2.0, CC-BY-NC 2.0, Public Domain, etc.). Metadata annotations are licensed under CC-BY 4.0. |
| **Open Images V7** | [storage.googleapis.com/openimages](https://storage.googleapis.com/openimages/web/index.html) | Safe & Dangerous | CC-BY 2.0 / CC-BY 4.0 | Yes (with credit) | **Attribution Caveat:** Image URLs link to Flickr photos under Creative Commons licenses. Authors must be credited as specified in the Open Images metadata table. |
| **Places365** | [places2.csail.mit.edu](http://places2.csail.mit.edu/) | Safe (Backgrounds) | MIT License | Research/Edu | Sourced from MIT Computer Vision Group. |
| **RVL-CDIP Subset** | [huggingface.co/datasets/rvl_cdip](https://huggingface.co/datasets/rvl_cdip) | Safe (Documents) | Open Access / Public | Research | Document image classification benchmark (clean business documents). |
| **RICO Subset** | [interactionmining.org/rico](https://interactionmining.org/rico) | Safe (Screenshots) | Research License | Research | Mobile application UI screenshots used to represent benign noise. |
| **Kaggle / Roboflow Weapons** | Custom Manual Ingestion | Dangerous & Safe | Source-Specific | User-Audited | **Manual Verification:** Datasets added to `data/raw/dangerous_manual/` must specify their individual Kaggle/Roboflow license in this file before training. |

---

## ⚖️ Class Definitions & Boundary Policies

### 1. `DANGEROUS` (Class 1)
- **Included:** Visible real firearms (`Handgun`, `Rifle`, `Shotgun`, `Submachine gun`) and real combat/edged weapons (`Combat knife`, `Bowie knife`, `Dagger`, `Tactical blade`).
- **Excluded:** Fictional futuristic lasers, microscopic parts, weapons fully occluded (>90%).
- **Strict Prohibition:** Absolutely **NO** graphic gore, open wounds, war violence casualties, or explicit material.

### 2. `SAFE` / Hard Negatives (Class 0 - Target $\ge 15\%$ Hard Negatives)
- **Included:** Domestic everyday objects, nature scenery, screenshots, documents, and specifically **Hard Negatives**:
  - Kitchen knives, bread knives, chef knives, table cutlery.
  - Workshop tools: Utility knives (box cutters), axes/hatchets as wood-cutting tools, scissors, shears.
  - Recreational & Sporting equipment: Fencing foils, target archery bows, plastic toys, brightly colored water guns.
  - Decorative/museum swords and historical armor.

---

## 📊 Dataset Ingestion & Class Distribution Tracker

*(Updated automatically at the Phase 1 Quota Gate)*

```
Total Safe Images:        [PENDING INGESTION]
  - COCO val2017:         [PENDING]
  - Open Images Safe:     [PENDING]
  - Hard Negatives:       [PENDING]
  - Documents/UI:         [PENDING]

Total Dangerous Images:   [PENDING INGESTION]
  - Handguns:             [PENDING]
  - Rifles:               [PENDING]
  - Combat Knives/Daggers:[PENDING]
  - Manual Uploads:       [PENDING]

Imbalance Ratio:          [PENDING]
```
