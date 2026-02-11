"""
🎭 PIPELINE D'AUGMENTATION OPTIMISÉ - Version 3 classes
Classes: 1-20, 21-50, 51-100
"""

import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import shutil
import json
from pathlib import Path
from datetime import datetime

# ============================================================
#                      CONFIGURATION - 3 CLASSES
# ============================================================
class Config:
    """Configuration pour 3 classes d'âge"""
    
    # CHEMINS
    DATA_PATH = Path("data\images_organisees")
    DATASET_PATH = Path("data\dataset_augmente_3classes")
    
    # Dimensions 
    IMG_SIZE = (224, 224)
    
    # Répartition
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Facteurs d'augmentation pour 3 classes
    # Adaptez selon le nombre d'images par classe
    AUG_FACTORS = {
        '1-20': 4,      # Ajustez selon vos données
        '21-50': 1,     # Ajustez selon vos données
        '51-100': 4,    # Ajustez selon vos données
    }
    
    # Seuils de qualité ajustés
    QUALITY_THRESHOLDS = {
        'min_sharpness': 50.0,
        'max_sharpness': 15000.0,
        'min_brightness': 15.0,
        'max_brightness': 240.0,
        'min_contrast': 20.0
    }
    
    # Classes d'âge (3 seulement)
    CLASSES = ['1-20', '21-50', '51-100']
    
    # Seed pour reproductibilité
    SEED = 42

np.random.seed(Config.SEED)
random.seed(Config.SEED)

# ============================================================
#           FONCTIONS D'ANALYSE DE QUALITÉ
# ============================================================
def calculate_sharpness(image):
    """Calcule la netteté (variance de Laplacien)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    """Calcule la luminosité moyenne"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,2])

def calculate_contrast(image):
    """Calcule le contraste (écart-type)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def check_image_quality(image_path):
    """Vérifie la qualité d'une image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            return False, "Image vide ou corrompue"
        
        sharpness = calculate_sharpness(img)
        brightness = calculate_brightness(img)
        contrast = calculate_contrast(img)
        
        if sharpness < Config.QUALITY_THRESHOLDS['min_sharpness']:
            return False, f"Netteté faible: {sharpness:.1f}"
        if sharpness > Config.QUALITY_THRESHOLDS['max_sharpness']:
            return False, f"Bruit excessif: {sharpness:.1f}"
        if brightness < Config.QUALITY_THRESHOLDS['min_brightness']:
            return False, f"Trop sombre: {brightness:.1f}"
        if brightness > Config.QUALITY_THRESHOLDS['max_brightness']:
            return False, f"Trop claire: {brightness:.1f}"
        if contrast < Config.QUALITY_THRESHOLDS['min_contrast']:
            return False, f"Contraste faible: {contrast:.1f}"
        
        return True, "OK"
    except Exception as e:
        return False, f"Erreur: {str(e)}"

# ============================================================
#       TRANSFORMATIONS
# ============================================================
def get_base_transform():
    """Transformation de base (resize seulement) pour val/test"""
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
    ])

def get_train_augmentations(class_name):
    """
    Augmentations pour le train UNIQUEMENT
    Adaptées aux 3 classes
    """
    
    # Augmentations standards pour toutes les classes
    # Vous pouvez différencier si une classe a des problèmes spécifiques
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(1, 3), p=0.2),
        A.GaussNoise(var_limit=(5, 15), p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.3
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.3
        ),
    ])

# ============================================================
#            COLLECTE DES IMAGES
# ============================================================
def collect_images_with_quality_check():
    """Collecte les images avec vérification de qualité"""
    images = []
    labels = []
    quality_stats = {cls: {'total': 0, 'accepted': 0, 'rejected': 0} 
                     for cls in Config.CLASSES}
    
    print("🔍 Collecte et vérification des images...")
    print(f"   Classes recherchées: {', '.join(Config.CLASSES)}")
    
    for class_name in Config.CLASSES:
        class_path = Config.DATA_PATH / class_name
        
        if not class_path.exists():
            print(f"⚠️  Dossier manquant: {class_path}")
            continue
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in class_path.glob(ext):
                quality_stats[class_name]['total'] += 1
                
                is_valid, reason = check_image_quality(img_path)
                
                if is_valid:
                    images.append(img_path)
                    labels.append(class_name)
                    quality_stats[class_name]['accepted'] += 1
                else:
                    quality_stats[class_name]['rejected'] += 1
        
        accepted = quality_stats[class_name]['accepted']
        rejected = quality_stats[class_name]['rejected']
        total = quality_stats[class_name]['total']
        
        if total > 0:
            accept_rate = (accepted / total) * 100
            print(f"   ✅ {class_name:8s}: {accepted:4d} acceptées / {total:4d} total ({accept_rate:.1f}%) | {rejected} rejetées")
        else:
            print(f"   ⚠️  {class_name:8s}: Aucune image trouvée")
    
    total_accepted = sum(stats['accepted'] for stats in quality_stats.values())
    total_all = sum(stats['total'] for stats in quality_stats.values())
    
    if total_all > 0:
        print(f"\n📊 TOTAL: {total_accepted} images valides / {total_all} totales ({(total_accepted/total_all)*100:.1f}%)")
    
    return images, labels, quality_stats

# ============================================================
#            CRÉATION DES SPLITS
# ============================================================
def create_balanced_splits(images, labels):
    """
    Split AVANT l'augmentation (évite le data leakage)
    """
    print("\n⚖️  Création des splits (avant augmentation)...")
    
    # Grouper par classe
    class_images = {cls: [] for cls in Config.CLASSES}
    for img, label in zip(images, labels):
        class_images[label].append(img)
    
    # Afficher la distribution
    print("\n📊 Distribution par classe:")
    for cls in Config.CLASSES:
        print(f"   {cls:8s}: {len(class_images[cls]):4d} images")
    
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    for cls in Config.CLASSES:
        imgs = class_images[cls]
        
        if len(imgs) == 0:
            print(f"⚠️  Classe {cls}: AUCUNE image!")
            continue
        
        if len(imgs) < 5:
            print(f"⚠️  Classe {cls}: seulement {len(imgs)} images - toutes mises dans train")
            X_train.extend(imgs)
            y_train.extend([cls] * len(imgs))
            continue
        
        # Split en train et temp (val+test)
        train_imgs, temp_imgs = train_test_split(
            imgs, 
            test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
            random_state=Config.SEED
        )
        
        # Split temp en val et test
        if len(temp_imgs) >= 2:
            val_imgs, test_imgs = train_test_split(
                temp_imgs,
                test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
                random_state=Config.SEED
            )
        else:
            # Si trop peu d'images, mettre au moins 1 dans val et 1 dans test
            val_imgs = temp_imgs[:1]
            test_imgs = temp_imgs[1:] if len(temp_imgs) > 1 else temp_imgs[:1]
        
        X_train.extend(train_imgs)
        y_train.extend([cls] * len(train_imgs))
        
        X_val.extend(val_imgs)
        y_val.extend([cls] * len(val_imgs))
        
        X_test.extend(test_imgs)
        y_test.extend([cls] * len(test_imgs))
        
        print(f"   {cls:8s}: Train={len(train_imgs):3d} | Val={len(val_imgs):3d} | Test={len(test_imgs):3d}")
    
    print(f"\n📊 Totaux:")
    print(f"   Train: {len(X_train):4d} images (seront augmentées)")
    print(f"   Val:   {len(X_val):4d} images (originales uniquement)")
    print(f"   Test:  {len(X_test):4d} images (originales uniquement)")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

# ============================================================
#            TRAITEMENT DES SPLITS
# ============================================================
def process_train_split(image_paths, labels):
    """Traite le split TRAIN avec augmentation"""
    print("\n🔄 Traitement TRAIN (avec augmentation)...")
    
    output_dir = Config.DATASET_PATH / 'train'
    total_saved = 0
    
    class_stats = {}
    
    for class_name in Config.CLASSES:
        class_indices = [i for i, label in enumerate(labels) if label == class_name]
        if not class_indices:
            continue
        
        class_images = [image_paths[i] for i in class_indices]
        num_aug = Config.AUG_FACTORS.get(class_name, 2)
        
        expected_total = len(class_images) * num_aug
        print(f"   🎯 {class_name:8s}: {len(class_images):3d} images × {num_aug} = {expected_total} total")
        
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        augmenter = get_train_augmentations(class_name)
        
        saved_count = 0
        for img_idx, img_path in enumerate(tqdm(class_images, desc=f"      {class_name}")):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Générer les versions augmentées
                for aug_idx in range(num_aug):
                    augmented = augmenter(image=img_rgb)
                    aug_img = augmented['image']
                    
                    # Normaliser et sauvegarder
                    if aug_img.dtype != np.uint8:
                        if aug_img.max() <= 1.0:
                            aug_img = (aug_img * 255).astype(np.uint8)
                        else:
                            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
                    
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    
                    save_name = f"{img_path.stem}_{img_idx:04d}_aug{aug_idx:02d}.jpg"
                    save_path = class_dir / save_name
                    
                    success = cv2.imwrite(str(save_path), aug_img_bgr, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        saved_count += 1
                        total_saved += 1
                    
            except Exception as e:
                print(f"      ⚠️  Erreur sur {img_path.name}: {e}")
                continue
        
        class_stats[class_name] = saved_count
        print(f"      ✅ {saved_count} images sauvegardées")
    
    print(f"\n   ✅ TRAIN TOTAL: {total_saved} images")
    return total_saved, class_stats

def process_val_test_split(split_name, image_paths, labels):
    """Traite VAL/TEST SANS augmentation (images originales)"""
    print(f"\n🔄 Traitement {split_name.upper()} (sans augmentation)...")
    
    output_dir = Config.DATASET_PATH / split_name
    total_saved = 0
    
    base_transform = get_base_transform()
    
    class_stats = {}
    
    for class_name in Config.CLASSES:
        class_indices = [i for i, label in enumerate(labels) if label == class_name]
        if not class_indices:
            continue
        
        class_images = [image_paths[i] for i in class_indices]
        print(f"   🎯 {class_name:8s}: {len(class_images)} images")
        
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for img_idx, img_path in enumerate(tqdm(class_images, desc=f"      {class_name}")):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Seulement resize, pas d'augmentation
                transformed = base_transform(image=img_rgb)
                trans_img = transformed['image']
                
                if trans_img.dtype != np.uint8:
                    if trans_img.max() <= 1.0:
                        trans_img = (trans_img * 255).astype(np.uint8)
                    else:
                        trans_img = np.clip(trans_img, 0, 255).astype(np.uint8)
                
                trans_img_bgr = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
                
                save_name = f"{img_path.stem}_{img_idx:04d}.jpg"
                save_path = class_dir / save_name
                
                success = cv2.imwrite(str(save_path), trans_img_bgr, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    saved_count += 1
                    total_saved += 1
                
            except Exception as e:
                print(f"      ⚠️  Erreur sur {img_path.name}: {e}")
                continue
        
        class_stats[class_name] = saved_count
        print(f"      ✅ {saved_count} images sauvegardées")
    
    print(f"\n   ✅ {split_name.upper()} TOTAL: {total_saved} images")
    return total_saved, class_stats

# ============================================================
#            FONCTION PRINCIPALE
# ============================================================
def main():
    """Pipeline principal"""
    
    print("\n" + "="*70)
    print("🚀 PIPELINE D'AUGMENTATION - 3 CLASSES D'ÂGE")
    print("="*70)
    print(f"📅 Classes: {', '.join(Config.CLASSES)}")
    print(f"📏 Taille images: {Config.IMG_SIZE}")
    print(f"📊 Ratio splits: Train={Config.TRAIN_RATIO} | Val={Config.VAL_RATIO} | Test={Config.TEST_RATIO}")
    print("✅ Pas de data leakage (split avant augmentation)")
    print("✅ Augmentation uniquement sur train")
    print("="*70)
    
    # Créer structure
    if Config.DATASET_PATH.exists():
        print("\n🧹 Nettoyage de l'ancien dataset...")
        shutil.rmtree(Config.DATASET_PATH)
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    # 1. Collecter toutes les images
    images, labels, quality_stats = collect_images_with_quality_check()
    
    if not images:
        print("\n❌ ERREUR: Aucune image valide trouvée!")
        print("   Vérifiez que les dossiers existent:")
        for cls in Config.CLASSES:
            print(f"   - {Config.DATA_PATH / cls}")
        return False
    
    # 2. Split AVANT augmentation
    splits = create_balanced_splits(images, labels)
    
    # 3. Traiter train avec augmentation
    train_count, train_stats = process_train_split(*splits['train'])
    
    # 4. Traiter val/test SANS augmentation
    val_count, val_stats = process_val_test_split('val', *splits['val'])
    test_count, test_stats = process_val_test_split('test', *splits['test'])
    
    # 5. Rapport final détaillé
    print("\n" + "="*70)
    print("📊 RAPPORT FINAL")
    print("="*70)
    
    print("\n🎯 RÉSUMÉ PAR SPLIT:")
    print(f"   Train: {train_count:5d} images (augmentées)")
    print(f"   Val:   {val_count:5d} images (originales)")
    print(f"   Test:  {test_count:5d} images (originales)")
    print(f"   TOTAL: {train_count + val_count + test_count:5d} images")
    
    print("\n📈 DISTRIBUTION PAR CLASSE:")
    print(f"   {'Classe':<10} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("   " + "-"*60)
    for cls in Config.CLASSES:
        t = train_stats.get(cls, 0)
        v = val_stats.get(cls, 0)
        te = test_stats.get(cls, 0)
        tot = t + v + te
        print(f"   {cls:<10} | {t:<8} | {v:<8} | {te:<8} | {tot:<8}")
    
    # Sauvegarder config
    config_data = {
        'classes': Config.CLASSES,
        'image_size': Config.IMG_SIZE,
        'split_ratios': {
            'train': Config.TRAIN_RATIO,
            'val': Config.VAL_RATIO,
            'test': Config.TEST_RATIO
        },
        'augmentation_factors': Config.AUG_FACTORS,
        'quality_thresholds': Config.QUALITY_THRESHOLDS,
        'statistics': {
            'train': {'total': train_count, 'per_class': train_stats},
            'val': {'total': val_count, 'per_class': val_stats},
            'test': {'total': test_count, 'per_class': test_stats}
        },
        'note': 'Split effectué AVANT augmentation pour éviter le data leakage',
        'created_at': datetime.now().isoformat()
    }
    
    config_file = Config.DATASET_PATH / 'config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Configuration sauvegardée: {config_file}")
    
    print("\n" + "="*70)
    print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"📁 Dataset créé: {Config.DATASET_PATH}")
    print("\n📝 PROCHAINES ÉTAPES:")
    print("   1. Vérifiez les images générées")
    print("   2. Ajustez les facteurs d'augmentation si nécessaire")
    print("   3. Lancez l'entraînement avec normalisation ImageNet")
    print("   4. Surveillez les courbes train/val pour l'overfitting")
    
    return True

# ============================================================
#            POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    try:
        import albumentations
        import cv2
        import sklearn
        print("✅ Toutes les dépendances sont installées\n")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("\n📦 Installation requise:")
        print("pip install albumentations opencv-python scikit-learn tqdm pandas")
        exit(1)
    
    success = main()
    
    if not success:
        print("\n❌ Le pipeline a rencontré des problèmes")
        exit(1)