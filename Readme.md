# 🎭 Age Classification with Face Masks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📋 Description

Projet de classification d'âge et de détection de masques faciaux utilisant des techniques de Deep Learning. Ce système permet de classifier automatiquement l'âge des personnes et de détecter la présence de masques sur des images de visages.

### 🎯 Objectifs du projet

- Classifier l'âge en 3 catégories : 1-20, 21-50, 51-100
- Détecter la présence ou l'absence de masques faciaux
- Fournir une application web interactive pour la prédiction en temps réel

### ✨ Fonctionnalités

- 🤖 Modèles de Deep Learning multiples (ResNet50, MobileNetV2, EfficientNet, DenseNet, ViT)
- 🎨 Application web Flask pour l'inférence
- 📊 Visualisation des résultats et métriques
- 🔄 Pipeline d'augmentation de données
- 📈 Notebooks d'entraînement détaillés

---

## 📁 Structure du projet

```
age-classification-masks/
├── 📂 data/                          # ⚠️ NON inclus dans le repo (voir ci-dessous)
│   ├── 1-20/                        # Images 1-20 ans
│   ├── 21-50/                       # Images 21-50 ans
│   └── 51-100/                      # Images 51-100 ans
│
├── 📂 src/
│   ├── 📂 app/                      # Application Flask
│   │   ├── app.py                   # Point d'entrée de l'application
│   │   ├── 📂 models/               # Configurations des modèles
│   │   ├── 📂 static/               # CSS, JS, assets
│   │   └── 📂 templates/            # Templates HTML
│   │
│   ├── 📂 Models/                   # Notebooks d'entraînement
│   │   ├── vit_transformer.ipynb    # Vision Transformer
│   │   ├── densenet.ipynb           # DenseNet
│   │   ├── EfficientNetB0.ipynb     # EfficientNet
│   │   ├── MobileNetV2.ipynb        # MobileNet V2
│   │   ├── model_resnet50.py        # ResNet50
│   │   └── model_svm.py             # SVM (baseline)
│   │
│   ├── app_data.py                  # Gestion des données pour l'app
│   ├── aug_data.py                  # Augmentation de données
│   └── predict_simple.py            # Script de prédiction simple
│
├── 📂 models/                       # Modèles entraînés (configurations)
│
├── 📄 requirements.txt              # Dépendances Python
├── 📄 Rapport_projet_ia.docx        # Rapport détaillé du projet
├── 📄 .gitignore                    # Fichiers ignorés par Git
└── 📄 README.md                     # Ce fichier
```

---

## ⚙️ Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git

### Étapes d'installation

1. **Cloner le repository**

```bash
git clone https://github.com/SARA-MAGGAG/age-classification-masks.git
cd age-classification-masks
```

2. **Créer un environnement virtuel**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Télécharger les données (voir section ci-dessous)**

---

## 📊 Données

### ⚠️ Important : Les données ne sont PAS incluses dans ce repository

En raison de leur taille volumineuse (~2GB), les données d'entraînement ne sont pas hébergées sur GitHub.

### Structure des données attendue

```
data/
├── 1-20/           # Images de personnes âgées de 1 à 20 ans
├── 21-50/          # Images de personnes âgées de 21 à 50 ans
└── 51-100/         # Images de personnes âgées de 51 à 100 ans
```

### 📥 Comment obtenir les données ?

**Option 1 : Télécharger depuis le lien**
```
🔗 Lien Google Drive : https://drive.google.com/drive/folders/1jq1UuRkLjtd_LzOJi2I8zGmSLs_CQQYs?usp=drive_link
```

**Option 2 : Contacter l'auteur**
```
📧 Email : saramaggag@gmail.com
```

**Option 3 : Utiliser vos propres données**

Organisez vos images selon la structure ci-dessus et placez-les dans le dossier `data/`

---

## 🚀 Utilisation

### 1. Organiser les données

Si vous avez des images brutes, utilisez le script d'organisation :

```bash
python src/aug_data.py
```

### 2. Entraîner un modèle

Ouvrez et exécutez l'un des notebooks dans `src/Models/` :

```bash
jupyter notebook src/Models/vit_transformer.ipynb
```

### 3. Lancer l'application web

```bash
cd src/app
python app.py
```

Accédez à l'application sur : `http://localhost:5000`

### 4. Faire des prédictions en ligne de commande

```bash
python src/predict_simple.py --image chemin/vers/image.jpg
```

---

## 🤖 Modèles disponibles

| Modèle | Architecture | Accuracy | Notebook |
|--------|-------------|----------|----------|
| **ViT** | Vision Transformer | 🥇 Best | `vit_transformer.ipynb` |
| **DenseNet** | DenseNet-121 | 🥈 | `densenet.ipynb` |
| **EfficientNet** | EfficientNetB0 | 🥉 | `EfficientNetB0.ipynb` |
| **MobileNetV2** | MobileNetV2 | ⚡ Fast | `MobileNetV2.ipynb` |
| **ResNet50** | ResNet-50 | 📊 | `model_resnet50.py` |
| **SVM** | Support Vector Machine | 📉 Baseline | `model_svm.py` |

---

## 📈 Résultats

Les résultats détaillés, métriques et visualisations sont disponibles dans :

- 📄 **Rapport complet** : `Rapport_projet_ia.docx`
- 📊 **Notebooks** : Chaque notebook contient ses propres visualisations
- 🗂️ **Modèles entraînés** : Disponibles sur demande (trop volumineux pour GitHub)

---

## 🛠️ Technologies utilisées

### Deep Learning & ML
- **TensorFlow / Keras** - Framework principal
- **PyTorch** - Pour certains modèles (ViT)
- **scikit-learn** - Métriques et preprocessing
- **OpenCV** - Traitement d'images

### Application Web
- **Flask** - Framework web
- **Bootstrap** - Interface utilisateur
- **JavaScript** - Interactions frontend

### Data Science
- **NumPy** - Calculs numériques
- **Pandas** - Manipulation de données
- **Matplotlib / Seaborn** - Visualisations

---

## 📝 Catégories d'âge

| Code | Tranche d'âge | Description |
|------|---------------|-------------|
| `1-20` | 1 à 20 ans | Enfants et adolescents |
| `21-50` | 21 à 50 ans | Adultes |
| `51-100` | 51 à 100 ans | Seniors |

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request




## 👥 Auteur

**Sara MAGGAG**

- 🔗 GitHub: [@SARA-MAGGAG](https://github.com/SARA-MAGGAG)
- 📧 Email: saramaggag@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/sara-maggag-a376661b7/

---

## 📚 Références

- Vision Transformer (ViT): [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- EfficientNet: [Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- DenseNet: [Densely Connected Networks](https://arxiv.org/abs/1608.06993)

---

## ⚠️ Notes importantes

1. **Données volumineuses** : Le dossier `data/` et les modèles entraînés ne sont pas inclus dans le repo GitHub
2. **Ressources GPU** : L'entraînement des modèles nécessite idéalement un GPU
3. **Versions** : Vérifiez la compatibilité des versions dans `requirements.txt`

---

## 🔄 Mises à jour

- **v1.0** (Février 2026) - Version initiale avec 6 modèles et application web

---

<div align="center">
  
**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

</div>
