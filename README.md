# 🔍 Facial Recognition — Biometric Systems

**UV SAFE — IMT Nord Europe**  
Yanis Ghazi · Amar Lassal  
Supervisor: Pauline Puteaux, LIRMM CNRS

---

## Overview

This project compares three face recognition approaches, from classical methods to deep learning:

| Method | Dataset | Accuracy | Errors |
|---|---|---|---|
| **LBPH + Chi²** | AT&T (40 subjects, 400 imgs) | 90.0% | 8/80 |
| **Eigenfaces + SVM** | AT&T (40 subjects, 400 imgs) | 96.25% | 3/80 |
| **FaceNet + SVM** | LFW (1140 identities, 3000+ imgs) | 98.61% | 12/865 |

---

## Project Structure

```
├── notebooks/
│   ├── 01_LBPH_Eigenfaces_ATT.ipynb   # Classical methods on AT&T
│   └── 02_FaceNet_LFW.ipynb           # Deep learning on LFW
└── README.md
```

---

## Methods

### 1. LBPH (Local Binary Patterns Histograms)
- For each pixel, compares its neighbours → local binary code
- Splits the image into blocks → LBP histogram per block → descriptor vector
- Classification via **Chi² distance** (nearest neighbour)
- Key hyperparameter: `block_size=8` (optimal), Chi² > Euclidean > KL
- **Macro AUC = 0.967**, with no supervised training

### 2. Eigenfaces + SVM
- Flatten 92×112px image → 10,304-dimensional vector
- **PCA (20 components)** → 70.5% variance retained, 10,304 → 20 dims
- **RBF SVM** classifier (`C=10`, `γ=scale`, `whiten=True`)
- Sensitive to global lighting, requires facial alignment
- Inference time: ~0.006 s

### 3. FaceNet (Google, 2015)
- **InceptionResNet** architecture pre-trained on VGGFace2 (3.3M images, 9,131 identities)
- Detection + alignment via **MTCNN**
- 512D embedding + **linear SVM**
- Triplet loss: same-person images pulled together, different-person images pushed apart
- Errors concentrate around low confidence scores (< 0.5)

---

## Datasets

### AT&T Database of Faces
- 400 images · 40 subjects × 10 images
- Resolution 92×112px, PGM format, 256 grayscale levels
- Split: **8 train / 2 test** per subject
- Variations: pose, lighting, expression, glasses, time
- [Kaggle](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

### LFW (Labeled Faces in the Wild)
- ~13,000 images · 5,000+ people (uncontrolled web photos)
- Filtered subset: 1,140 individuals, 3,000+ images (≥10 imgs/person)
- [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

---

## Installation

```bash
pip install numpy pillow matplotlib seaborn pandas scikit-learn scikit-image
```

For the FaceNet notebook (GPU recommended):

```bash
pip install torch facenet-pytorch
```

---

## Usage

The notebooks are designed to run on **Kaggle** with the corresponding datasets.

1. Import the dataset on Kaggle
2. Open the corresponding notebook
3. *Run All*

---

## Key Results

- **Shared errors** across both classical methods: subjects S5, S10, S28 — intrinsic visual ambiguity that neither LBPH nor Eigenfaces can resolve
- **FaceNet** reaches 98.61% on LFW, a significantly harder dataset than AT&T, with no network retraining (transfer learning)
- A **confidence-based rejection mechanism** (SVM score < 0.5) would eliminate most FaceNet false positives

---

## References

- Samaria & Harter (1994) — AT&T Database of Faces
- Turk & Pentland (1991) — Eigenfaces for Recognition
- Ojala et al. (2002) — Local Binary Patterns
- Schroff et al. (2015) — FaceNet
- Deng et al. (2019) — ArcFace (state of the art: 99.8% LFW)
