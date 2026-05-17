# 🧬 Projet Deep Learning — Classification Histologique sur PathMNIST

> **DU Sorbonne Data Analytics**
> Date : Mars 2026

---

## 📋 Table des matières

- [Présentation](#présentation)
- [Technologies utilisées](#technologies-utilisées)
- [Dataset : PathMNIST](#dataset--pathmnist)
- [Structure du projet](#structure-du-projet)
- [Pipeline de modélisation](#pipeline-de-modélisation)
- [Résultats](#résultats)
- [Lancer le projet](#lancer-le-projet)

---

## Présentation

Ce projet explore la **classification automatique d'images histologiques** à partir du dataset **PathMNIST**, en comparant plusieurs architectures de deep learning :

- **MLP (Dense Network)** — baseline simple
- **CNN** — avec et sans augmentation de données
- **Transfer Learning** — ResNet-18 pré-entraîné sur ImageNet
- **Vision Transformer (ViT)** — mécanisme d'attention global
- **Grad-CAM** — interprétabilité des décisions du modèle

L'objectif est de classer correctement des images de tissus colorectaux en **9 catégories** (Adipose, Background, Debris, Lymphocytes, Mucus, Muscle, Normal Colon Mucosa, Cancer-Associated Stroma, Colorectal Adenocarcinoma Epithelium), avec une application potentielle en aide au diagnostic clinique.

---

## Technologies utilisées

| Outil | Rôle |
|---|---|
| [Python 3](https://www.python.org/) | Langage principal |
| [PyTorch](https://pytorch.org/) | Framework deep learning |
| [TorchVision](https://pytorch.org/vision/) | Modèles pré-entraînés (ResNet-18) |
| [MedMNIST](https://medmnist.com/) | Chargement du dataset PathMNIST |
| [NumPy](https://numpy.org/) | Calculs numériques |
| [Matplotlib](https://matplotlib.org/) | Visualisations |
| [Google Colab](https://colab.research.google.com/) | Entraînement sur GPU (T4) |

---

## Dataset : PathMNIST

Le dataset **PathMNIST** est un jeu de données d'images histologiques colorectales issues de tissus médicaux.

| Caractéristique | Valeur |
|---|---|
| Taille des images | 28×28 pixels (RGB) |
| Nombre de classes | 9 |
| Taille train | ~90 000 images |
| Taille validation | ~10 000 images |
| Taille test | ~7 000 images |
| Distribution | Équilibrée (~10 000 images par classe en train) |

### Classes

| Classe | Description |
|---|---|
| Adipose | Tissu adipeux |
| Background | Fond / arrière-plan |
| Debris | Débris cellulaires |
| Lymphocytes | Cellules immunitaires |
| Mucus | Mucus |
| Smooth Muscle | Muscle lisse |
| Normal Colon Mucosa | Muqueuse colique normale |
| Cancer-Associated Stroma | Stroma associé au cancer |
| Colorectal Adenocarcinoma Epithelium | Épithélium adénocarcinome colorectal |

> ⚠️ Le fichier de modèle `model_aug_pathmnist.pth` n'est **pas inclus** dans ce dépôt en raison de sa taille. Relance l'entraînement via le notebook `3_CNN2.ipynb` pour le régénérer.

---

## Structure du projet

```
📁 dlprojets vf/
│
├── 1_DataExploration.ipynb               # Exploration et analyse du dataset
├── 2_Dense_Netweork_Baseline.ipynb       # MLP baseline (Dense Network)
├── 3_CNN2.ipynb                          # CNN avec/sans augmentation de données
├── 4_TransferLearning.ipynb              # Transfer Learning (ResNet-18)
├── 5_VisionTransformer.ipynb             # Vision Transformer (ViT)
├── 6_Grad-CAM.ipynb                      # Interprétabilité Grad-CAM
├── Final Comparison and Analysis.ipynb   # Comparaison finale de tous les modèles
├── model_aug_pathmnist.pth               # ⚠️ Poids du CNN (non inclus dans le repo)
└── .gitignore
```

---

## Pipeline de modélisation

### Notebook 1 — Exploration des données
- Chargement du dataset PathMNIST via `medmnist`
- Visualisation des 9 classes
- Analyse de la distribution (dataset équilibré)
- Comparaison visuelle Background vs Debris (texture uniforme vs hétérogène)

---

### Notebook 2 — MLP Baseline (Dense Network)
- Architecture entièrement connectée (fully connected)
- Entraînement sur 15 époques
- **Résultat : 62.90% de précision sur le test set**
- Analyse de la matrice de confusion (confusion principale : Smooth Muscle → Adipose)

---

### Notebook 3 — CNN avec augmentation de données
- Architecture CNN à 3 blocs convolutionnels avec BatchNorm et Dropout
- Entraînement sur 40 époques sans puis avec augmentation de données
- **Résultat : > 75% de précision sur le test set**
- Analyse de l'overfitting et de l'effet du Dropout

---

### Notebook 4 — Transfer Learning (ResNet-18)
Deux expériences comparées :

| Expérience | Stratégie | Précision test |
|---|---|---|
| **A — Frozen** | Couches gelées, seule la tête est entraînée | **85.79%** |
| **B — Full Fine-Tuning** | Tous les poids sont mis à jour (lr=0.0001) | **86.02%** |

- Utilisation des poids officiels `ResNet18_Weights.IMAGENET1K_V1`
- Discussion sur les effets de l'upscaling 28×28 → 224×224
- Analyse des raisons du succès du Transfer Learning (features universelles de bas niveau)

---

### Notebook 5 — Vision Transformer (ViT)
- Implémentation d'un ViT from scratch (**552 585 paramètres**)
- Analyse de l'effet de la taille des patches (7×7 vs 14×14)
- Analyse de l'impact des positional embeddings
- Le ViT est moins performant que le CNN sur ce dataset (manque de données, petites images)

---

### Notebook 6 — Grad-CAM (Interprétabilité)
- Implémentation de Grad-CAM avec hooks PyTorch
- Visualisation des heatmaps sur la dernière couche convolutionnelle
- Analyse des zones d'activation sur des exemples correctement et mal classés
- Exemple de mauvaise prédiction : Colorectal Adenocarcinoma prédit comme Lymphocytes

---

### Notebook Final — Comparaison et Analyse

| Modèle | Précision test | Remarques |
|---|---|---|
| MLP Baseline | 62.90% | Ignore la structure spatiale |
| CNN | > 75% | Meilleur compromis performance / complexité |
| ResNet-18 Frozen | 85.79% | Transfer Learning efficace |
| ResNet-18 Full | **86.02%** | Meilleure performance globale |
| Vision Transformer | < CNN | Nécessite plus de données |

**Conclusion :** Le CNN (avec Transfer Learning complet) est le modèle recommandé pour un éventuel déploiement clinique, combiné à Grad-CAM pour l'interprétabilité.

---

## Lancer le projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/rishikaran21/Projet-Deep-Learning.git
cd Projet-Deep-Learning
```

### 2. Installer les dépendances

```bash
pip install torch torchvision medmnist numpy matplotlib
```

### 3. Exécuter les notebooks dans l'ordre

```
1_DataExploration.ipynb
2_Dense_Netweork_Baseline.ipynb
3_CNN2.ipynb                     ← génère model_aug_pathmnist.pth
4_TransferLearning.ipynb
5_VisionTransformer.ipynb
6_Grad-CAM.ipynb                 ← nécessite model_aug_pathmnist.pth
Final Comparison and Analysis.ipynb
```

> 💡 L'entraînement a été réalisé sur **Google Colab avec GPU T4**. Il est fortement recommandé d'utiliser un GPU pour reproduire les résultats dans des temps raisonnables.

---

*Projet réalisé dans le cadre du DU Sorbonne Data Analytics — Mars 2026*
