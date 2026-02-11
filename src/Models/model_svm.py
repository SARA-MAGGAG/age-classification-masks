"""
🎭 SVM 100% CLASSIQUE - Classification d'Âge (3 Classes)
Méthodes classiques uniquement : HOG, LBP, Histogrammes
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                      CONFIGURATION OPTIMISÉE
# ============================================================
class SVMConfig:
    """Configuration optimisée pour SVM sur dataset augmenté"""
    
    # Chemins
    DATASET_PATH = Path("data\dataset_augmente_3classes")
    
    # LIMITATION CRUCIALE : échantillonnage pour éviter surcharge
    MAX_SAMPLES_PER_CLASS = {
        'train': 1000,   # Maximum d'images par classe pour train
        'val': None,     # Prendre toutes les images val
        'test': None     # Prendre toutes les images test
    }
    
    # Extraction de caractéristiques CLASSIQUES uniquement
    FEATURE_TYPE = 'hog'  # Options: 'hog', 'lbp', 'histogram'
    # 'hog' = Recommandé (meilleur compromis vitesse/performance)
    # 'lbp' = Plus rapide, moins précis
    # 'histogram' = Le plus rapide, basique
    IMG_SIZE = (128, 128)
    
    # PCA
    USE_PCA = True
    PCA_COMPONENTS = 100
    
    # SVM - Paramètres restreints pour GridSearch rapide
    PARAM_GRID = {
        'C': [1, 10],
        'gamma': ['scale', 0.001],
        'kernel': ['rbf']
    }
    
    # Gestion du déséquilibre
    USE_CLASS_WEIGHT = True
    
    # Seed
    SEED = 42

np.random.seed(SVMConfig.SEED)

# ============================================================
#          EXTRACTION DE CARACTÉRISTIQUES
# ============================================================
def extract_hog_features(image):
    """Extrait HOG"""
    from skimage.feature import hog
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, SVMConfig.IMG_SIZE)
    
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True,
                   feature_vector=True)
    
    return features

def extract_lbp_features(image):
    """Extrait LBP"""
    from skimage.feature import local_binary_pattern
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, SVMConfig.IMG_SIZE)
    
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, 'uniform')
    
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    
    return hist

def extract_color_histogram(image):
    """Extrait histogramme couleur"""
    image_resized = cv2.resize(image, SVMConfig.IMG_SIZE)
    
    hist_b = cv2.calcHist([image_resized], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image_resized], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image_resized], [2], None, [32], [0, 256])
    
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    
    hist_features = np.hstack([hist_b, hist_g, hist_r])
    
    return hist_features

# ============================================================
#          CHARGEMENT AVEC ÉCHANTILLONNAGE
# ============================================================
def load_dataset_sampled(split='train'):
    """
    Charge avec échantillonnage stratifié pour éviter surcharge mémoire
    """
    split_path = SVMConfig.DATASET_PATH / split
    max_samples = SVMConfig.MAX_SAMPLES_PER_CLASS.get(split)
    
    images = []
    labels = []
    
    print(f"\n📂 Chargement {split} (échantillonné)...")
    
    for class_name in sorted(os.listdir(split_path)):
        class_path = split_path / class_name
        
        if not class_path.is_dir():
            continue
        
        # Lister toutes les images
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        # ÉCHANTILLONNAGE si nécessaire
        if max_samples and len(image_files) > max_samples:
            print(f"   ⚠️  {class_name}: {len(image_files)} images → échantillonné à {max_samples}")
            image_files = np.random.choice(image_files, max_samples, replace=False)
        
        print(f"   📁 {class_name}: chargement de {len(image_files)} images...", end=" ")
        
        class_images = []
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    class_images.append(img)
            except Exception as e:
                continue
        
        images.extend(class_images)
        labels.extend([class_name] * len(class_images))
        
        print(f"✅ {len(class_images)} chargées")
    
    print(f"📊 Total {split}: {len(images)} images")
    return images, labels

# ============================================================
#          EXTRACTION AVEC GESTION D'ERREURS
# ============================================================
def extract_features_safe(images, feature_type='hog'):
    """Extrait features CLASSIQUES avec gestion d'erreurs robuste"""
    print(f"\n🔍 Extraction des caractéristiques ({feature_type})...")
    
    # Vérifier que c'est bien une méthode classique
    if feature_type not in ['hog', 'lbp', 'histogram']:
        raise ValueError(f"Méthode non classique: {feature_type}. Utilisez 'hog', 'lbp' ou 'histogram'")
    
    features_list = []
    failed = 0
    
    for i, img in enumerate(tqdm(images, desc=f"   {feature_type.upper()}")):
        try:
            if feature_type == 'hog':
                feat = extract_hog_features(img)
            elif feature_type == 'lbp':
                feat = extract_lbp_features(img)
            elif feature_type == 'histogram':
                feat = extract_color_histogram(img)
            
            features_list.append(feat)
            
        except Exception as e:
            failed += 1
            # Ajouter vecteur de zéros
            if features_list:
                features_list.append(np.zeros_like(features_list[0]))
            else:
                # Première image, on skip
                continue
    
    if failed > 0:
        print(f"   ⚠️  {failed} images échouées (remplacées par zéros)")
    
    return np.array(features_list)

# ============================================================
#          ENTRAÎNEMENT SVM OPTIMISÉ
# ============================================================
def train_svm_optimized(X_train, y_train, X_val=None, y_val=None):
    """Entraîne SVM avec optimisations"""
    print("\n🤖 Entraînement SVM optimisé...")
    
    # Encoder labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    if X_val is not None:
        y_val_encoded = le.transform(y_val)
    
    # Normalisation
    print("   📏 Normalisation...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    # PCA si activé
    pca = None
    if SVMConfig.USE_PCA and X_train_scaled.shape[1] > SVMConfig.PCA_COMPONENTS:
        print(f"   📉 PCA ({SVMConfig.PCA_COMPONENTS} composantes)...")
        pca = PCA(n_components=SVMConfig.PCA_COMPONENTS, random_state=SVMConfig.SEED)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"      Variance expliquée: {variance_explained:.2%}")
        
        if X_val is not None:
            X_val_scaled = pca.transform(X_val_scaled)
    
    # Créer SVM avec class_weight
    print("   🔍 GridSearch avec validation croisée...")
    
    svm_base = SVC(
        random_state=SVMConfig.SEED,
        probability=True,
        class_weight='balanced' if SVMConfig.USE_CLASS_WEIGHT else None
    )
    
    # GridSearch RESTREINT
    grid_search = GridSearchCV(
        svm_base,
        SVMConfig.PARAM_GRID,
        cv=3,  # 3-fold au lieu de 5 pour aller plus vite
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )
    
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train_encoded)
    elapsed = time.time() - start_time
    
    print(f"\n   ✅ GridSearch terminé en {elapsed/60:.1f} minutes")
    print(f"   🎯 Meilleurs params: {grid_search.best_params_}")
    print(f"   📊 Score CV: {grid_search.best_score_:.4f}")
    
    # Évaluation sur validation
    best_model = grid_search.best_estimator_
    
    if X_val is not None:
        val_pred = best_model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val_encoded, val_pred)
        print(f"\n   📈 Accuracy validation: {val_acc:.4f}")
        
        print("\n   📋 Rapport validation:")
        print(classification_report(y_val_encoded, val_pred, 
                                   target_names=le.classes_, 
                                   digits=4))
    
    return {
        'model': best_model,
        'scaler': scaler,
        'pca': pca,
        'label_encoder': le,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'grid_results': pd.DataFrame(grid_search.cv_results_)
    }

# ============================================================
#          ÉVALUATION
# ============================================================
def evaluate_model(model_dict, X_test, y_test):
    """Évalue le modèle sur test"""
    print("\n🧪 Évaluation sur le jeu de test...")
    
    model = model_dict['model']
    scaler = model_dict['scaler']
    pca = model_dict['pca']
    le = model_dict['label_encoder']
    
    # Préparer test
    X_test_scaled = scaler.transform(X_test)
    if pca:
        X_test_scaled = pca.transform(X_test_scaled)
    
    y_test_encoded = le.transform(y_test)
    
    # Prédictions
    test_pred = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)
    
    test_acc = accuracy_score(y_test_encoded, test_pred)
    
    print(f"\n📊 Accuracy test: {test_acc:.4f}")
    
    print("\n📋 Rapport de classification (test):")
    print(classification_report(y_test_encoded, test_pred,
                               target_names=le.classes_,
                               digits=4))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test_encoded, test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Matrice de Confusion - SVM ({SVMConfig.FEATURE_TYPE.upper()})')
    plt.ylabel('Vrai label')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig('logs/confusion_matrix_svm_optimized.png', dpi=300)
    print("\n💾 Matrice sauvegardée: confusion_matrix_svm_optimized.png")
    plt.show()
    
    # Métriques par classe
    print("\n📊 Métriques détaillées par classe:")
    for i, class_name in enumerate(le.classes_):
        class_acc = np.mean(test_pred[y_test_encoded == i] == i)
        print(f"   {class_name}: {class_acc:.4f}")
    
    return {
        'accuracy': test_acc,
        'predictions': test_pred,
        'probabilities': test_proba,
        'confusion_matrix': cm
    }

# ============================================================
#          FONCTION PRINCIPALE
# ============================================================
def main_svm_optimized():
    """Pipeline SVM optimisé"""
    
    print("\n" + "="*70)
    print("🤖 SVM 100% CLASSIQUE - CLASSIFICATION D'ÂGE (3 CLASSES)")
    print("="*70)
    print(f"📁 Dataset: {SVMConfig.DATASET_PATH}")
    print(f"🔍 Features CLASSIQUES: {SVMConfig.FEATURE_TYPE.upper()}")
    print(f"📊 Échantillonnage train: {SVMConfig.MAX_SAMPLES_PER_CLASS['train']} par classe")
    print(f"⚖️  Class weight: {'Activé' if SVMConfig.USE_CLASS_WEIGHT else 'Désactivé'}")
    print("="*70)
    
    # Vérifier dataset
    if not SVMConfig.DATASET_PATH.exists():
        print(f"\n❌ Dataset introuvable: {SVMConfig.DATASET_PATH}")
        print("   Exécutez d'abord le pipeline d'augmentation.")
        return False
    
    # 1. Charger avec échantillonnage
    print("\n" + "="*70)
    print("📥 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("="*70)
    
    X_train_img, y_train = load_dataset_sampled('train')
    X_val_img, y_val = load_dataset_sampled('val')
    X_test_img, y_test = load_dataset_sampled('test')
    
    # Distribution
    print("\n📊 Distribution:")
    for split, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   {split:5s}: {dict(zip(unique, counts))}")
    
    # 2. Extraire features
    print("\n" + "="*70)
    print("🔧 ÉTAPE 2: EXTRACTION DES CARACTÉRISTIQUES")
    print("="*70)
    
    X_train = extract_features_safe(X_train_img, SVMConfig.FEATURE_TYPE)
    X_val = extract_features_safe(X_val_img, SVMConfig.FEATURE_TYPE)
    X_test = extract_features_safe(X_test_img, SVMConfig.FEATURE_TYPE)
    
    print(f"\n✅ Shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # 3. Entraîner
    print("\n" + "="*70)
    print("🚀 ÉTAPE 3: ENTRAÎNEMENT SVM")
    print("="*70)
    
    model_dict = train_svm_optimized(X_train, y_train, X_val, y_val)
    
    # 4. Évaluer
    print("\n" + "="*70)
    print("🧪 ÉTAPE 4: ÉVALUATION FINALE")
    print("="*70)
    
    results = evaluate_model(model_dict, X_test, y_test)
    
    # 5. Sauvegarder
    print("\n" + "="*70)
    print("💾 ÉTAPE 5: SAUVEGARDE")
    print("="*70)
    
    save_path = Path("models")
    save_path.mkdir(exist_ok=True)
    
    joblib.dump(model_dict['model'], save_path / 'svm_model.pkl')
    joblib.dump(model_dict['scaler'], save_path / 'scaler.pkl')
    if model_dict['pca']:
        joblib.dump(model_dict['pca'], save_path / 'pca.pkl')
    joblib.dump(model_dict['label_encoder'], save_path / 'label_encoder.pkl')
    
    # Config
    import json
    config_save = {
        'feature_type': SVMConfig.FEATURE_TYPE,
        'img_size': SVMConfig.IMG_SIZE,
        'max_samples_per_class': SVMConfig.MAX_SAMPLES_PER_CLASS,
        'use_pca': SVMConfig.USE_PCA,
        'pca_components': SVMConfig.PCA_COMPONENTS,
        'use_class_weight': SVMConfig.USE_CLASS_WEIGHT,
        'best_params': model_dict['best_params'],
        'cv_score': float(model_dict['cv_score']),
        'test_accuracy': float(results['accuracy']),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print(f"   ✅ Modèle sauvegardé: {save_path}")
    
    # 6. Rapport final
    print("\n" + "="*70)
    print("📈 RAPPORT FINAL")
    print("="*70)
    print(f"🎯 Features: {SVMConfig.FEATURE_TYPE}")
    print(f"🔧 Params: {model_dict['best_params']}")
    print(f"📊 CV Score: {model_dict['cv_score']:.4f}")
    print(f"🧪 Test Accuracy: {results['accuracy']:.4f}")
    print("="*70)
    
    print("\n✅ Pipeline SVM terminé avec succès!")
    
    return True

# ============================================================
#          POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    # Vérifier dépendances
    try:
        import cv2
        import sklearn
        import matplotlib
        from skimage.feature import hog
        print("✅ Dépendances installées\n")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("\n📦 Installation:")
        print("pip install scikit-learn opencv-python matplotlib seaborn joblib scikit-image tqdm")
        exit(1)
    
    success = main_svm_optimized()
    
    if not success:
        exit(1)