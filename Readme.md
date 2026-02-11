AGE CLASSIFICATION MASKS PROJECT
================================

Description:
------------
Ce projet de classification d'âge et de détection de masques organise et classe 
les images de visages selon l'âge et la présence de masques.

Structure du projet:
-------------------
AGE_CLASSIFICATION_MASKS/
├── data/
│   ├── images/
│       ├── no_mask/
│       │   ├── nomask_001_018/
│       │   ├── nomask_019_065/
│       │   ├── nomask_066_110/
│       │   └── nomask_111_plus/
│       └── with_mask/
│          ├── mask_001_018/
│          ├── mask_019_065/
│          ├── mask_066_110/
│          └── mask_111_plus/
│   
└── src/
    ├── Models/
    ├── app_data.py
    ├── data.py
    ├── .gitignore
    ├── Readme.txt
    └── requirements.txt

Installation:
-------------
1. Cloner le repository
2. Créer un environnement virtuel:
   python -m venv venv
3. Activer l'environnement virtuel:
   - Windows: venv\Scripts\activate
4. Installer les dépendances:
   pip install -r requirements.txt

Utilisation:
------------
1. Organiser les images:
   Exécuter le script data.py pour organiser les images dans la structure appropriée:
   python src/data.py


Groupes d'âge:
--------------
- 001_018: 1 à 18 ans
- 019_065: 19 à 65 ans  
- 066_110: 66 à 110 ans
- 111_plus: Plus de 110 ans

Dépendances:
------------
Voir le fichier requirements.txt pour la liste complète des dépendances Python.

