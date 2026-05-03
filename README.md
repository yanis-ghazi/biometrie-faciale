#  Systèmes Biométriques de Reconnaissance Faciale

**UV SAFE — IMT Nord Europe**  
Yanis Ghazi · Amar Lassal  
Encadrante : Pauline Puteaux, LIRMM CNRS

---

##  Présentation

Ce projet compare trois approches de reconnaissance faciale, des méthodes classiques au deep learning :

| Méthode | Dataset | Accuracy | Erreurs |
|---|---|---|---|
| **LBPH + Chi²** | AT&T (40 sujets, 400 imgs) | 90.0% | 8/80 |
| **Eigenfaces + SVM** | AT&T (40 sujets, 400 imgs) | 96.25% | 3/80 |
| **FaceNet + SVM** | LFW (1140 identités, 3000+ imgs) | 98.61% | 12/865 |

---

##  Structure du projet

```
├── notebooks/
│   ├── 01_LBPH_Eigenfaces_ATT.ipynb   # Méthodes classiques sur AT&T
│   └── 02_FaceNet_LFW.ipynb           # Deep learning sur LFW
└── README.md
```

---

##  Méthodes

### 1. LBPH (Local Binary Patterns Histograms)
- Pour chaque pixel, compare ses voisins → code binaire local
- Découpe l'image en blocs → histogramme LBP par bloc → vecteur descripteur
- Classification par **distance Chi²** (nearest neighbour)
- Hyperparamètre clé : `block_size=8` (optimal), distance Chi² > Euclidienne > KL
- **Macro AUC = 0.967**, sans aucun entraînement supervisé

### 2. Eigenfaces + SVM
- Aplatissement 92×112px → vecteur 10 304 dimensions
- **PCA (20 composantes)** → 70.5% de variance conservée, 10 304 → 20 dims
- Classificateur **SVM RBF** (`C=10`, `γ=scale`, `whiten=True`)
- Sensible à l'éclairage global, nécessite alignement facial
- Temps d'inférence : ~0.006 s

### 3. FaceNet (Google, 2015)
- Architecture **InceptionResNet** pré-entraîné sur VGGFace2 (3.3M images, 9131 identités)
- Détection + alignement par **MTCNN**
- Embedding 512D + **SVM linéaire**
- Triplet loss : images du même individu proches, individus différents éloignés
- Les erreurs se concentrent sur les faibles scores de confiance (< 0.5)

---

##  Datasets

### AT&T Database of Faces
- 400 images · 40 sujets × 10 images
- Résolution 92×112px, format PGM, 256 niveaux de gris
- Split : **8 train / 2 test** par sujet
- Variations : pose, éclairage, expression, lunettes, temps
- [Kaggle](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

### LFW (Labeled Faces in the Wild)
- ~13 000 images · 5 000+ personnes (photos web non contrôlées)
- Sous-ensemble filtré : 1 140 individus, 3 000+ images (≥10 imgs/personne)
- [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

---

##  Installation

```bash
pip install numpy pillow matplotlib seaborn pandas scikit-learn scikit-image
```

Pour le notebook FaceNet (GPU recommandé) :

```bash
pip install torch facenet-pytorch
```

---

##  Utilisation

Les notebooks sont conçus pour tourner sur **Kaggle** avec les datasets correspondants.

1. Importer le dataset sur Kaggle
2. Ouvrir le notebook correspondant
3. *Run All*

---

##  Résultats clés

- **Erreurs communes** aux deux méthodes classiques : sujets S5, S10, S28 — ambiguïté visuelle intrinsèque que ni LBPH ni Eigenfaces ne parviennent à lever
- **FaceNet** atteint 98.61% sur LFW, un dataset nettement plus difficile qu'AT&T, sans ré-entraînement du réseau (transfert learning)
- Un mécanisme de **rejet par seuil de confiance** (score SVM < 0.5) permettrait d'éliminer la majorité des faux positifs FaceNet

---

##  Références

- Samaria & Harter (1994) — AT&T Database of Faces
- Turk & Pentland (1991) — Eigenfaces for Recognition
- Ojala et al. (2002) — Local Binary Patterns
- Schroff et al. (2015) — FaceNet
- Deng et al. (2019) — ArcFace (état de l'art : 99.8% LFW)
