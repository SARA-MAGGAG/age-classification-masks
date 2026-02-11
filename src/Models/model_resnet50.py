"""
Script ResNet50 ANTI-OVERFITTING AVANCÉ avec génération de courbes complètes
Incluant: ROC curves, Precision/Recall/F1 par classe, courbes d'entraînement
"""

import os
import sys
import time
import numpy as np
import json
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, CSVLogger, TerminateOnNaN, Callback)

import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import pandas as pd

# ========== CONFIGURATION AVANCÉE ==========
DATASET_PATH = "data\dataset_augmente_3classes"
MODEL_SAVE_PATH = "models"
LOG_DIR = "logs"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
INITIAL_LR = 0.0001
SEED = 42

# Hyperparamètres avancés
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
GRADIENT_CLIP = 1.0

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ========== UTILITAIRES ==========
def print_header(text):
    print("\n" + "="*60)
    print(f"🎯 {text}")
    print("="*60)

def print_success(text):
    print(f"✅ {text}")

def print_info(text):
    print(f"📊 {text}")

def print_warning(text):
    print(f"⚠️  {text}")

# ========== CALLBACK TIMING ==========
class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.total_start = None
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def on_train_begin(self, logs=None):
        self.total_start = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        diff = train_acc - val_acc
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1
        
        if (epoch + 1) % 2 == 0:
            print_info(f"Epoch {epoch+1:02d} | Time: {epoch_time:.1f}s")
            print_info(f"  Train: {train_acc:.2%} (loss: {train_loss:.4f})")
            print_info(f"  Val:   {val_acc:.2%} (loss: {val_loss:.4f})")
            print_info(f"  Gap:   {diff:.2%} {'⚠️' if diff > 0.05 else '✅'}")
    
    def on_train_end(self, logs=None):
        if self.total_start:
            total_time = time.time() - self.total_start
            print_success(f"Meilleur val_accuracy: {self.best_val_acc:.2%} (epoch {self.best_epoch})")
            print_info(f"Temps total: {total_time/60:.1f} min")

class PeriodicCheckpoint(Callback):
    def __init__(self, filepath, save_freq=5, save_best_only=False, monitor='val_accuracy'):
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best = -np.inf if 'acc' in monitor else np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    print_warning(f"Metric {self.monitor} not found")
                    return
                    
                if 'acc' in self.monitor:
                    if current > self.best:
                        self.best = current
                        filepath = self.filepath.format(epoch=epoch+1)
                        self.model.save(filepath)
                        print_info(f"Model saved to {filepath}")
                else:
                    if current < self.best:
                        self.best = current
                        filepath = self.filepath.format(epoch=epoch+1)
                        self.model.save(filepath)
                        print_info(f"Model saved to {filepath}")
            else:
                filepath = self.filepath.format(epoch=epoch+1)
                self.model.save(filepath)
                print_info(f"Model saved to {filepath}")

# ========== FONCTIONS EXISTANTES (gardées telles quelles) ==========
def categorical_crossentropy_with_label_smoothing(y_true, y_pred, smoothing=LABEL_SMOOTHING):
    y_true = y_true * (1.0 - smoothing) + smoothing / tf.cast(tf.shape(y_pred)[-1], tf.float32)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def create_advanced_augmentations():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1],
        channel_shift_range=10.0,
        fill_mode='reflect'
    )

def load_data():
    print_header("CHARGEMENT DES DONNÉES AVANCÉ")
    
    train_datagen = create_advanced_augmentations()
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    print_info("Chargement du dataset d'entraînement...")
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    print_info("Chargement du dataset de validation...")
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'val'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print_info("Chargement du dataset de test...")
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)
    
    print_success(f"Classes: {num_classes} ({', '.join(class_names)})")
    print_info(f"Train: {train_gen.samples:,} images")
    print_info(f"Val: {val_gen.samples:,} images")
    print_info(f"Test: {test_gen.samples:,} images")
    
    return train_gen, val_gen, test_gen, class_names, num_classes

def build_advanced_model(num_classes):
    print_header("CONSTRUCTION DU MODÈLE AVANCÉ")
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IMAGE_SIZE + (3,)
    )
    
    base_model.trainable = False
    for layer in base_model.layers[-40:]:
        layer.trainable = True
    
    trainable_count = sum([l.trainable for l in base_model.layers])
    total_count = len(base_model.layers)
    
    print_info(f"Couches totales: {total_count}")
    print_info(f"Couches entraînables: {trainable_count} ({trainable_count/total_count:.1%})")
    
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01),
              bias_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(
        learning_rate=INITIAL_LR,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=GRADIENT_CLIP
    )
    
    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy_with_label_smoothing,
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    print_success("Modèle construit avec succès")
    print_info(f"Paramètres totaux: {model.count_params():,}")
    
    return model

def create_advanced_callbacks():
    callbacks = [
        TimingCallback(),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='min',
            min_delta=0.005
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
            cooldown=2,
            min_delta=0.001
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'resnet50_best_advanced.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        PeriodicCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'resnet50_epoch_{epoch:02d}.keras'),
            save_freq=5,
            save_best_only=False
        ),
        CSVLogger(
            filename=os.path.join(LOG_DIR, 'training_history_advanced.csv'),
            separator=',',
            append=False
        ),
        TerminateOnNaN()
    ]
    
    return callbacks

def train_advanced_model(model, train_gen, val_gen):
    print_header("ENTRAÎNEMENT AVANCÉ")
    
    callbacks = create_advanced_callbacks()
    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)
    
    print_info(f"Steps par epoch: {steps_per_epoch}")
    print_info(f"Validation steps: {validation_steps}")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ========== NOUVELLES FONCTIONS DE VISUALISATION COMPLÈTES ==========

def plot_training_curves(history):
    """
    Génère les courbes d'entraînement comme dans votre image:
    - Courbe de Loss (Train et Val)
    - Courbe d'Accuracy (Train et Val)
    """
    print_header("GÉNÉRATION DES COURBES D'ENTRAÎNEMENT")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(len(history.history['loss']))
    
    # Courbe de Loss
    ax1.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history.history['val_loss'], 'orange', linewidth=2, label='Val Loss')
    ax1.axvline(x=14, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Début fine-tuning')
    ax1.set_xlabel('Époches', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Courbe de perte (Loss)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Courbe d'Accuracy
    ax2.plot(epochs, history.history['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, history.history['val_accuracy'], 'orange', linewidth=2, label='Val Accuracy')
    ax2.axvline(x=14, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Début fine-tuning')
    ax2.set_xlabel('Époches', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Précision (Accuracy)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"Courbes d'entraînement sauvegardées: {save_path}")

def plot_metrics_by_age_class(y_true, y_pred, y_pred_proba, class_names):
    """
    Génère le graphique en barres: Précision, Recall et F1-score par classe d'âge
    Comme dans votre Figure 4.2
    """
    print_header("GÉNÉRATION DES MÉTRIQUES PAR CLASSE")
    
    # Calculer precision, recall, f1 pour chaque classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Créer le graphique en barres
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Précision', 
                   color='#365E8E', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   color='#EB7D34', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, f1, width, label='F1-score', 
                   color='#DC5343', edgecolor='black', linewidth=0.5)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Classes', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Précision, Recall et F1-score par classe d\'âge', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, 'metrics_by_class.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"Métriques par classe sauvegardées: {save_path}")
    
    # Afficher les résultats
    print("\n📊 MÉTRIQUES PAR CLASSE:")
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Précision: {precision[i]:.2%}")
        print(f"  Recall:    {recall[i]:.2%}")
        print(f"  F1-score:  {f1[i]:.2%}")
        print(f"  Support:   {support[i]}")

def plot_roc_curves(y_true, y_pred_proba, class_names, num_classes):
    """
    Génère les courbes ROC pour chaque classe (one-vs-rest)
    """
    print_header("GÉNÉRATION DES COURBES ROC")
    
    # Binariser les labels pour ROC multiclasse
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Calculer ROC et AUC pour chaque classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculer micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Tracer les courbes ROC pour chaque classe
    for i, color in zip(range(num_classes), colors[:num_classes]):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Tracer la courbe micro-average
    ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
           label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    # Ligne diagonale (classificateur aléatoire)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Hasard (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=11)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=11)
    ax.set_title('Courbes ROC - Classification multiclasse', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, 'roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"Courbes ROC sauvegardées: {save_path}")
    
    # Afficher les AUC
    print("\n📊 SCORES AUC PAR CLASSE:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {roc_auc[i]:.4f}")
    print(f"  Micro-average: {roc_auc['micro']:.4f}")

def plot_confusion_matrix(cm, class_names):
    """
    Génère une matrice de confusion visuellement attractive
    """
    print_header("GÉNÉRATION DE LA MATRICE DE CONFUSION")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normaliser la matrice de confusion
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Classe Prédite',
           ylabel='Classe Réelle',
           title='Matrice de Confusion Normalisée')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Ajouter les valeurs dans chaque cellule
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"Matrice de confusion sauvegardée: {save_path}")

def generate_complete_visualizations(model, test_gen, class_names, num_classes, history):
    """
    Fonction principale qui génère TOUTES les visualisations
    """
    print_header("GÉNÉRATION COMPLÈTE DES VISUALISATIONS")
    
    # 1. Courbes d'entraînement (Loss et Accuracy)
    plot_training_curves(history)
    
    # 2. Obtenir les prédictions
    print_info("Calcul des prédictions sur le test set...")
    test_gen.reset()
    test_steps = int(np.ceil(test_gen.samples / BATCH_SIZE))
    
    y_pred_proba = model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes[:len(y_pred)]  # S'assurer que les tailles correspondent
    
    # 3. Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)
    
    # 4. Métriques par classe (barres)
    plot_metrics_by_age_class(y_true, y_pred, y_pred_proba, class_names)
    
    # 5. Courbes ROC
    plot_roc_curves(y_true, y_pred_proba, class_names, num_classes)
    
    print_success("\n✅ TOUTES LES VISUALISATIONS ONT ÉTÉ GÉNÉRÉES!")
    print_info(f"Dossier de sauvegarde: {LOG_DIR}")
    print_info("Fichiers générés:")
    print_info("  - training_curves.png (Loss et Accuracy)")
    print_info("  - metrics_by_class.png (Précision/Recall/F1)")
    print_info("  - roc_curves.png (Courbes ROC)")
    print_info("  - confusion_matrix.png (Matrice de confusion)")

# ========== FONCTION D'ÉVALUATION MODIFIÉE ==========
def evaluate_advanced_model(model, test_gen, class_names):
    print_header("ÉVALUATION AVANCÉE")
    
    test_gen.reset()
    test_steps = int(np.ceil(test_gen.samples / BATCH_SIZE))
    
    print_info(f"Évaluation sur {test_gen.samples:,} images")
    
    results = model.evaluate(test_gen, steps=test_steps, verbose=1, return_dict=True)
    
    print_success("RÉSULTATS DE L'ÉVALUATION:")
    for metric, value in results.items():
        if 'accuracy' in metric.lower() or 'precision' in metric.lower() or 'recall' in metric.lower():
            print(f"  {metric}: {value:.2%}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    test_gen.reset()
    y_true = test_gen.classes
    predictions = model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print_header("RAPPORT DE CLASSIFICATION")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    return results.get('accuracy', 0), cm

def save_advanced_model(model, class_names, test_accuracy, confusion_matrix):
    print_header("SAUVEGARDE AVANCÉE")
    
    model_path = os.path.join(MODEL_SAVE_PATH, 'resnet50_advanced_final.h5')
    model.save(model_path)
    print_success(f"Modèle sauvegardé: {model_path}")
    
    metadata = {
        'model': {
            'name': 'ResNet50_Advanced_AntiOverfitting',
            'architecture': 'ResNet50 + Custom Head',
            'input_size': IMAGE_SIZE,
            'num_classes': len(class_names),
            'total_params': int(model.count_params())
        },
        'training': {
            'initial_learning_rate': INITIAL_LR,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'label_smoothing': LABEL_SMOOTHING,
            'mixup_alpha': MIXUP_ALPHA,
            'gradient_clip': GRADIENT_CLIP
        },
        'performance': {
            'test_accuracy': float(test_accuracy),
            'best_model_path': 'resnet50_best_advanced.keras'
        },
        'dataset': {
            'classes': class_names
        },
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(MODEL_SAVE_PATH, 'metadata_advanced.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print_success(f"Métadonnées sauvegardées: {metadata_path}")

# ========== MAIN MODIFIÉ ==========
def main():
    print("\n" + "="*80)
    print("🚀 RESNET50 AVEC GÉNÉRATION COMPLÈTE DE COURBES")
    print("="*80)
    
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print_success(f"GPU disponible: {len(gpus)}")
    else:
        print_warning("Pas de GPU disponible - utilisation du CPU")
    
    try:
        # 1. Charger les données
        train_gen, val_gen, test_gen, class_names, num_classes = load_data()
        
        # 2. Construire le modèle
        model = build_advanced_model(num_classes)
        model.summary()
        
        # 3. Entraîner
        history = train_advanced_model(model, train_gen, val_gen)
        
        # 4. Évaluer
        test_accuracy, cm = evaluate_advanced_model(model, test_gen, class_names)
        
        # 5. *** GÉNÉRATION DE TOUTES LES VISUALISATIONS ***
        generate_complete_visualizations(model, test_gen, class_names, num_classes, history)
        
        # 6. Sauvegarder
        save_advanced_model(model, class_names, test_accuracy, cm)
        
        # 7. Résumé final
        print_header("🎉 RÉSUMÉ FINAL 🎉")
        print_success(f"Test Accuracy: {test_accuracy:.2%}")
        print_info(f"Tous les graphiques sont dans: {LOG_DIR}")
        print_info("Graphiques générés:")
        print_info("  ✅ training_curves.png - Courbes Loss et Accuracy")
        print_info("  ✅ metrics_by_class.png - Précision/Recall/F1 par classe")
        print_info("  ✅ roc_curves.png - Courbes ROC multiclasse")
        print_info("  ✅ confusion_matrix.png - Matrice de confusion")
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT ET VISUALISATIONS TERMINÉS AVEC SUCCÈS")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()