import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from flask import Flask, Response, jsonify, request, render_template
import json
import threading
import time
from datetime import datetime
import base64
from werkzeug.utils import secure_filename
import traceback
import pickle
import joblib
from skimage.feature import hog, local_binary_pattern

# Ajout des imports pour PyTorch/ViT
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# INITIALISATION FLASK
# ============================================================
app = Flask(__name__, template_folder='templates')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'js'), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024

# ============================================================
# VARIABLES GLOBALES
# ============================================================
model = None  # Pour les modèles TensorFlow
svm_model = None
scaler = None
pca = None
label_encoder = None
vit_model = None  # Pour Vision Transformer
vit_transform = None  # Transformations pour ViT
current_model_name = None
current_model_type = None  # 'resnet50', 'mobilenetv2', 'efficientnet', 'classic_svm', 'vit'

MODEL_CLASSES = ['1-20', '21-50', '51-100']
CLASS_MAPPING = {'1-20': 'young', '21-50': 'adult', '51-100': 'senior'}
AGE_RANGES = {'young': '1-20', 'adult': '21-50', 'senior': '51-100'}
IMG_SIZE = (224, 224)

cam = None
camera_active = False
face_cascade = None

detection_stats = {
    'frames_processed': 0,
    'faces_detected': 0,
    'young_count': 0,
    'adult_count': 0,
    'senior_count': 0,
    'start_time': datetime.now().isoformat()
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_face_detector():
    """Initialise le détecteur de visages avec plusieurs cascades"""
    global face_cascade
    try:
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_alt_tree.xml'
        ]
        
        for cascade_file in cascade_files:
            try:
                cascade_path = cv2.data.haarcascades + cascade_file
                temp_cascade = cv2.CascadeClassifier(cascade_path)
                if not temp_cascade.empty():
                    face_cascade = temp_cascade
                    print(f"✅ Détecteur initialisé: {cascade_file}")
                    return True
            except:
                continue
        
        print("⚠️ Détecteur en mode dégradé")
        return False
    except Exception as e:
        print(f"❌ Erreur détecteur: {e}")
        return False

def list_available_models():
    """Liste tous les modèles disponibles"""
    models_info = []
    
    for filename in os.listdir(MODELS_DIR):
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Modèles TensorFlow/Keras
        if filename.endswith('.h5') or filename.endswith('.keras'):
            if 'resnet50' in filename.lower():
                model_type = 'ResNet50'
                description = 'Deep Learning avec ResNet50'
            elif 'mobilenet' in filename.lower():
                model_type = 'MobileNetV2'
                description = 'Deep Learning avec MobileNetV2'
            elif 'efficientnet' in filename.lower():
                model_type = 'EfficientNetB0'
                description = 'Deep Learning avec EfficientNetB0'
            elif 'efficient' in filename.lower():
                model_type = 'EfficientNet'
                description = 'Deep Learning avec EfficientNet'
            else:
                model_type = 'CNN Custom'
                description = 'Modèle CNN personnalisé'
            
            models_info.append({
                'filename': filename,
                'display_name': f'{model_type} - {filename}',
                'type': model_type,
                'description': description
            })
        
        # Modèle SVM Classique
        elif filename == 'svm_model.pkl':
            svm_files = ['svm_model.pkl', 'scaler.pkl', 'pca.pkl', 'label_encoder.pkl']
            svm_exists = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in svm_files)
            
            if svm_exists:
                models_info.append({
                    'filename': 'svm_model.pkl',
                    'display_name': 'SVM Classique',
                    'type': 'SVM Classique',
                    'description': 'Méthodes traditionnelles (HOG/LBP/Histogramme)'
                })
        
        # Modèle Vision Transformer (PyTorch)
        elif filename.endswith('.pth'):
            model_type = 'Vision Transformer'
            description = 'Vision Transformer (PyTorch) - Architecture avancée'
            
            models_info.append({
                'filename': filename,
                'display_name': f'Vision Transformer - {filename}',
                'type': model_type,
                'description': description
            })
    
    return models_info

def load_classic_svm():
    """Charge le pipeline SVM classique"""
    global svm_model, scaler, pca, label_encoder, current_model_type, config_data
    
    try:
        print(f"\n🔧 CHARGEMENT DU PIPELINE SVM CLASSIQUE")
        
        svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        pca_path = os.path.join(MODELS_DIR, 'pca.pkl')
        label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
        config_path = os.path.join(MODELS_DIR, 'config.json')
        
        try:
            svm_model = joblib.load(svm_path)
        except:
            with open(svm_path, 'rb') as f:
                svm_model = pickle.load(f)
        
        try:
            scaler = joblib.load(scaler_path)
        except:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        pca = None
        if os.path.exists(pca_path):
            try:
                pca = joblib.load(pca_path)
            except:
                with open(pca_path, 'rb') as f:
                    pca = pickle.load(f)
        
        try:
            label_encoder = joblib.load(label_encoder_path)
        except:
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        
        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        
        current_model_type = 'classic_svm'
        
        print(f"✅ SVM model chargé: {type(svm_model).__name__}")
        print(f"✅ Scaler chargé: {scaler.__class__.__name__}")
        if pca:
            print(f"✅ PCA chargé: {pca.__class__.__name__}")
        print(f"✅ Label encoder chargé: {label_encoder.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DE CHARGEMENT SVM CLASSIQUE: {e}")
        traceback.print_exc()
        return False

def load_resnet50_model(model_name):
    """Charge un modèle ResNet50"""
    global model, current_model_type
    
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        print(f"\n🔧 CHARGEMENT DU MODÈLE RESNET50: {model_name}")
        print(f" Chemin: {model_path}")
        print(f" Taille: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        current_model_type = 'resnet50'
        
        print(f"✅ MODÈLE RESNET50 CHARGÉ AVEC SUCCÈS\n")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DE CHARGEMENT: {e}")
        traceback.print_exc()
        return False

def load_mobilenetv2_model(model_name):
    """Charge un modèle MobileNetV2"""
    global model, current_model_type
    
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        print(f"\n🔧 CHARGEMENT DU MODÈLE MOBILENETV2: {model_name}")
        print(f" Chemin: {model_path}")
        print(f" Taille: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        current_model_type = 'mobilenetv2'
        
        print(f"✅ MODÈLE MOBILENETV2 CHARGÉ AVEC SUCCÈS\n")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DE CHARGEMENT MOBILENETV2: {e}")
        traceback.print_exc()
        return False

def load_efficientnet_model(model_name):
    """Charge un modèle EfficientNetB0"""
    global model, current_model_type
    
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        print(f"\n🔧 CHARGEMENT DU MODÈLE EFFICIENTNETB0: {model_name}")
        print(f" Chemin: {model_path}")
        print(f" Taille: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        current_model_type = 'efficientnet'
        
        print(f"✅ MODÈLE EFFICIENTNETB0 CHARGÉ AVEC SUCCÈS\n")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DE CHARGEMENT EFFICIENTNETB0: {e}")
        traceback.print_exc()
        return False

def load_vit_model(model_name):
    """Charge un modèle Vision Transformer (PyTorch)"""
    global vit_model, vit_transform, current_model_type
    
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        print(f"\n🔧 CHARGEMENT DU MODÈLE VISION TRANSFORMER: {model_name}")
        print(f" Chemin: {model_path}")
        
        # Charger le modèle PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️  Utilisation du device: {device}")
        
        # Charger le modèle
        vit_model = torch.load(model_path, map_location=device)
        vit_model.eval()  # Mode évaluation
        
        # Définir les transformations pour ViT
        vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        current_model_type = 'vit'
        
        print(f"✅ MODÈLE VISION TRANSFORMER CHARGÉ AVEC SUCCÈS")
        print(f"✅ Device: {device}")
        print(f"✅ Transformations: Resize(224x224), Normalize(ImageNet)\n")
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DE CHARGEMENT VISION TRANSFORMER: {e}")
        traceback.print_exc()
        return False

def load_model(model_name):
    """Charge le modèle sélectionné"""
    global current_model_name
    
    current_model_name = model_name
    
    # Réinitialiser les modèles précédents
    global model, svm_model, vit_model
    model = None
    svm_model = None
    vit_model = None
    
    # Déterminer le type de modèle et charger
    if model_name == 'svm_model.pkl':
        return load_classic_svm()
    elif 'mobilenet' in model_name.lower():
        return load_mobilenetv2_model(model_name)
    elif 'efficientnet' in model_name.lower() or 'efficient' in model_name.lower():
        return load_efficientnet_model(model_name)
    elif 'vit' in model_name.lower() or model_name.endswith('.pth'):
        return load_vit_model(model_name)
    elif 'resnet50' in model_name.lower():
        return load_resnet50_model(model_name)
    else:
        # Par défaut, essayer de charger comme ResNet50
        return load_resnet50_model(model_name)

# ============================================================
# CLASSE CAMÉRA
# ============================================================
class Camera:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.simulation = True
        self.init()
    
    def init(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.simulation = False
                    print("✅ Caméra détectée")
                    return True
        except Exception as e:
            print(f"⚠️ Erreur caméra: {e}")
        
        print("⚠️ Mode simulation")
        return False
    
    def get_frame(self):
        with self.lock:
            if self.cap and self.cap.isOpened() and not self.simulation:
                ret, frame = self.cap.read()
                if ret:
                    return frame
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for i in range(480):
                color = int(50 + 100 * i / 480)
                cv2.line(frame, (0, i), (640, i), (color, color, color), 1)
            
            cv2.putText(frame, "MODE SIMULATION", (200, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Model: {current_model_name or 'None'}", (200, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Type: {current_model_type or 'None'}", (200, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame
    
    def release(self):
        if self.cap:
            self.cap.release()

# ============================================================
# FONCTIONS DE PRÉDICTION
# ============================================================
def extract_hog_features(image):
    """Extrait les caractéristiques HOG"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       feature_vector=True)
        return features
    except Exception as e:
        print(f"⚠️ Erreur HOG: {e}")
        return None

def extract_lbp_features(image):
    """Extrait les caractéristiques LBP"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, 'uniform')
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist
    except Exception as e:
        print(f"⚠️ Erreur LBP: {e}")
        return None

def extract_color_histogram(image):
    """Extrait l'histogramme de couleur"""
    try:
        image_resized = cv2.resize(image, (128, 128))
        
        hist_b = cv2.calcHist([image_resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image_resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([image_resized], [2], None, [32], [0, 256])
        
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        
        hist_features = np.hstack([hist_b, hist_g, hist_r])
        return hist_features
    except Exception as e:
        print(f"⚠️ Erreur histogramme: {e}")
        return None

def predict_with_classic_svm(face_image):
    """Prédiction avec SVM classique"""
    try:
        feature_type = config_data.get('feature_type', 'hog') if config_data else 'hog'
        
        if feature_type == 'hog':
            features = extract_hog_features(face_image)
        elif feature_type == 'lbp':
            features = extract_lbp_features(face_image)
        elif feature_type == 'histogram':
            features = extract_color_histogram(face_image)
        else:
            features = extract_hog_features(face_image)
        
        if features is None:
            return None
        
        features = features.reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        if pca:
            features_scaled = pca.transform(features_scaled)
        
        if hasattr(svm_model, 'predict_proba'):
            probabilities = svm_model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(probabilities)
            confidence = float(probabilities[pred_idx])
        else:
            pred_idx = svm_model.predict(features_scaled)[0]
            confidence = 1.0
        
        if hasattr(label_encoder, 'inverse_transform'):
            try:
                pred_class = label_encoder.inverse_transform([pred_idx])[0]
            except:
                pred_class = str(pred_idx)
        else:
            pred_class = str(pred_idx)
        
        if pred_class in CLASS_MAPPING:
            interface_class = CLASS_MAPPING[pred_class]
        else:
            for key in CLASS_MAPPING:
                if key in pred_class or pred_class in key:
                    interface_class = CLASS_MAPPING[key]
                    break
            else:
                interface_class = 'adult'
        
        return {
            'model_class': pred_class,
            'interface_class': interface_class,
            'age_range': AGE_RANGES.get(interface_class, '21-50'),
            'confidence': confidence,
            'confidence_percent': int(confidence * 100)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur prédiction SVM classique: {e}")
        return None

def predict_with_resnet50(face_image):
    """Prédiction avec ResNet50"""
    try:
        face_resized = cv2.resize(face_image, IMG_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = np.expand_dims(face_rgb, axis=0)
        face_array = resnet_preprocess(face_array.astype(np.float32))
        
        predictions = model.predict(face_array, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx])
        model_class = MODEL_CLASSES[pred_idx]
        interface_class = CLASS_MAPPING.get(model_class, 'adult')
        
        return {
            'model_class': model_class,
            'interface_class': interface_class,
            'age_range': AGE_RANGES[interface_class],
            'confidence': confidence,
            'confidence_percent': int(confidence * 100)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur prédiction ResNet50: {e}")
        return None

def predict_with_mobilenetv2(face_image):
    """Prédiction avec MobileNetV2"""
    try:
        face_resized = cv2.resize(face_image, IMG_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = np.expand_dims(face_rgb, axis=0)
        face_array = mobilenet_preprocess(face_array.astype(np.float32))
        
        predictions = model.predict(face_array, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx])
        model_class = MODEL_CLASSES[pred_idx]
        interface_class = CLASS_MAPPING.get(model_class, 'adult')
        
        return {
            'model_class': model_class,
            'interface_class': interface_class,
            'age_range': AGE_RANGES[interface_class],
            'confidence': confidence,
            'confidence_percent': int(confidence * 100)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur prédiction MobileNetV2: {e}")
        return None

def predict_with_efficientnet(face_image):
    """Prédiction avec EfficientNetB0"""
    try:
        face_resized = cv2.resize(face_image, IMG_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = np.expand_dims(face_rgb, axis=0)
        face_array = efficientnet_preprocess(face_array.astype(np.float32))
        
        predictions = model.predict(face_array, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx])
        model_class = MODEL_CLASSES[pred_idx]
        interface_class = CLASS_MAPPING.get(model_class, 'adult')
        
        return {
            'model_class': model_class,
            'interface_class': interface_class,
            'age_range': AGE_RANGES[interface_class],
            'confidence': confidence,
            'confidence_percent': int(confidence * 100)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur prédiction EfficientNetB0: {e}")
        return None

def predict_with_vit(face_image):
    """Prédiction avec Vision Transformer"""
    try:
        # Convertir l'image OpenCV en PIL
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Appliquer les transformations
        input_tensor = vit_transform(pil_image).unsqueeze(0)
        
        # Déplacer sur le bon device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = vit_model(input_tensor)
            
            # Gérer différents formats de sortie
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Prendre le premier élément si c'est un tuple
            
            # Appliquer softmax pour obtenir les probabilités
            probabilities = F.softmax(outputs, dim=1)[0]
            
            pred_idx = torch.argmax(probabilities).item()
            confidence = float(probabilities[pred_idx])
        
        model_class = MODEL_CLASSES[pred_idx]
        interface_class = CLASS_MAPPING.get(model_class, 'adult')
        
        return {
            'model_class': model_class,
            'interface_class': interface_class,
            'age_range': AGE_RANGES[interface_class],
            'confidence': confidence,
            'confidence_percent': int(confidence * 100)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur prédiction Vision Transformer: {e}")
        traceback.print_exc()
        return None

def predict_age_category(face_image):
    """Fonction de prédiction principale"""
    if current_model_type == 'classic_svm':
        return predict_with_classic_svm(face_image)
    elif current_model_type == 'resnet50' and model is not None:
        return predict_with_resnet50(face_image)
    elif current_model_type == 'mobilenetv2' and model is not None:
        return predict_with_mobilenetv2(face_image)
    elif current_model_type == 'efficientnet' and model is not None:
        return predict_with_efficientnet(face_image)
    elif current_model_type == 'vit' and vit_model is not None:
        return predict_with_vit(face_image)
    else:
        print("⚠️ Aucun modèle chargé")
        return None

def detect_faces(image):
    """Détecte les visages avec plusieurs méthodes"""
    if face_cascade is None:
        return []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            return faces
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.02,
            minNeighbors=2,
            minSize=(20, 20)
        )
        
        if len(faces) > 0:
            return faces
        
        return []
        
    except Exception as e:
        print(f"⚠️ Erreur détection: {e}")
        return []

def draw_single_face_box(frame, x, y, w, h, result):
    """Dessine un seul cadre avec style pour la caméra"""
    color_map = {
        'young': (0, 165, 255),
        'adult': (0, 255, 0),
        'senior': (255, 0, 0)
    }
    
    color = color_map.get(result['interface_class'], (0, 255, 0))
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
    
    label = f"{result['interface_class'].upper()} ({result['age_range']})"
    
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    if y - text_h - 20 > 0:
        text_y = y - 10
        text_bg_y1 = text_y - text_h - 5
        text_bg_y2 = text_y + 5
    else:
        text_y = y + h + text_h + 10
        text_bg_y1 = text_y - text_h - 5
        text_bg_y2 = text_y + 5
    
    cv2.rectangle(
        frame, 
        (x, text_bg_y1), 
        (x + text_w, text_bg_y2), 
        color, 
        -1
    )
    
    cv2.putText(
        frame, label, (x, text_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    return frame

# ============================================================
# ROUTES FLASK
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = cam.get_frame()
                detection_stats['frames_processed'] += 1
                
                # Vérifier si un modèle est chargé et si le détecteur de visages est disponible
                if ((model is not None and current_model_type in ['resnet50', 'mobilenetv2', 'efficientnet']) or 
                    (svm_model is not None and current_model_type == 'classic_svm') or
                    (vit_model is not None and current_model_type == 'vit')) and face_cascade:
                    
                    faces = detect_faces(frame)
                    
                    if len(faces) > 0:
                        max_area = 0
                        main_face = None
                        
                        for (x, y, w, h) in faces:
                            area = w * h
                            if area > max_area:
                                max_area = area
                                main_face = (x, y, w, h)
                        
                        if main_face:
                            x, y, w, h = main_face
                            face_roi = frame[y:y+h, x:x+w]
                            result = predict_age_category(face_roi)
                            
                            if result:
                                detection_stats['faces_detected'] += 1
                                if result['interface_class'] == 'young':
                                    detection_stats['young_count'] += 1
                                elif result['interface_class'] == 'adult':
                                    detection_stats['adult_count'] += 1
                                else:
                                    detection_stats['senior_count'] += 1
                                
                                frame = draw_single_face_box(frame, x, y, w, h, result)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(0.033)
                
            except Exception as e:
                print(f"⚠️ Erreur dans video_feed: {e}")
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    rate = 0
    if detection_stats['frames_processed'] > 0:
        rate = int((detection_stats['faces_detected'] / detection_stats['frames_processed']) * 100)
    
    return jsonify({
        'success': True,
        'stats': {
            'frames_processed': detection_stats['frames_processed'],
            'faces_detected': detection_stats['faces_detected'],
            'detection_rate': rate,
            'young_count': detection_stats['young_count'],
            'adult_count': detection_stats['adult_count'],
            'senior_count': detection_stats['senior_count']
        }
    })

@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    global detection_stats
    detection_stats = {
        'frames_processed': 0,
        'faces_detected': 0,
        'young_count': 0,
        'adult_count': 0,
        'senior_count': 0,
        'start_time': datetime.now().isoformat()
    }
    return jsonify({'success': True})

@app.route('/api/models')
def get_models():
    models_info = list_available_models()
    return jsonify({
        'success': True,
        'models': models_info,
        'count': len(models_info)
    })

@app.route('/api/model_info')
def get_model_info():
    architecture = ""
    if current_model_type == 'classic_svm':
        architecture = "SVM Classique (HOG/LBP/Histogramme)"
    elif current_model_type == 'resnet50':
        architecture = "ResNet50 (Deep Learning)"
    elif current_model_type == 'mobilenetv2':
        architecture = "MobileNetV2 (Deep Learning léger)"
    elif current_model_type == 'efficientnet':
        architecture = "EfficientNetB0 (Deep Learning efficient)"
    elif current_model_type == 'vit':
        architecture = "Vision Transformer (PyTorch)"
    
    return jsonify({
        'success': True,
        'model_name': current_model_name,
        'model_type': current_model_type,
        'architecture': architecture
    })

@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    data = request.json
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'message': 'Nom de modèle non fourni'})
    
    if load_model(model_name):
        return jsonify({
            'success': True,
            'message': f'Modèle {model_name} chargé',
            'model_name': current_model_name,
            'model_type': current_model_type
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Erreur lors du chargement du modèle'
        })

@app.route('/api/analyze_upload', methods=['POST'])
def analyze_upload():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'Aucun fichier'})
    
    file = request.files['image']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Fichier invalide'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'message': 'Impossible de lire l\'image'})
        
        h, w = image.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        faces = detect_faces(image)
        
        if len(faces) == 0:
            print("⚠️ Aucun visage détecté, analyse de l'image entière")
            h, w = image.shape[:2]
            faces = np.array([[0, 0, w, h]])
        
        predictions = []
        
        for (x, y, w, h) in faces:
            margin_w, margin_h = int(w * 0.1), int(h * 0.1)
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(image.shape[1], x + w + margin_w)
            y2 = min(image.shape[0], y + h + margin_h)
            
            face_roi = image[y1:y2, x1:x2]
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue
            
            result = predict_age_category(face_roi)
            
            if result:
                detection_stats['faces_detected'] += 1
                if result['interface_class'] == 'young':
                    detection_stats['young_count'] += 1
                elif result['interface_class'] == 'adult':
                    detection_stats['adult_count'] += 1
                else:
                    detection_stats['senior_count'] += 1
                
                predictions.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'prediction': result
                })
        
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        message = f"{len(faces)} visage(s) détecté(s)"
        if len(faces) == 1 and faces[0][0] == 0 and faces[0][1] == 0:
            message = "Analyse globale de l'image"
        
        return jsonify({
            'success': True,
            'num_faces': len(faces),
            'predictions': predictions,
            'image_original': f"data:image/jpeg;base64,{img_str}",
            'message': message,
            'model_used': current_model_name,
            'model_type': current_model_type
        })
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

# ============================================================
# INITIALISATION
# ============================================================
def initialize_app():
    global cam
    print("\n" + "="*70)
    print("🚀 INITIALISATION DE L'APPLICATION")
    print("="*70)
    
    print(f"📁 Dossier app: {BASE_DIR}")
    print(f"📁 Dossier models: {MODELS_DIR}")
    print(f"📁 Dossier uploads: {UPLOAD_FOLDER}")
    
    if not os.path.exists(MODELS_DIR):
        print(f"\n❌ ERREUR: Le dossier models n'existe pas!")
        print(f" Chemin configuré: {MODELS_DIR}")
        return
    else:
        print(f"✅ Dossier models trouvé!")
    
    init_face_detector()
    
    models_info = list_available_models()
    print(f"\n📋 MODÈLES DISPONIBLES ({len(models_info)} fichiers):")
    
    if models_info:
        for i, model_info in enumerate(models_info, 1):
            print(f" {i}. {model_info['display_name']}")
            print(f" - Type: {model_info['type']}")
            print(f" - Description: {model_info['description']}")
        
        model_to_load = None
        
        # Priorité de chargement
        # 1. EfficientNetB0
        # 2. MobileNetV2
        # 3. Vision Transformer
        # 4. ResNet50
        # 5. SVM Classique
        # 6. Premier modèle disponible
        
        for model_info in models_info:
            if 'efficientnetb0' in model_info['filename'].lower() or 'efficient' in model_info['filename'].lower():
                model_to_load = model_info['filename']
                print(f"\n✅ MODÈLE efficientnetb0 TROUVÉ: {model_to_load}")
                print(f" → Utilisation de ce modèle par défaut (excellent rapport performance/rapidité)")
                break
        
        if not model_to_load:
            for model_info in models_info:
                if 'mobilenet' in model_info['filename'].lower():
                    model_to_load = model_info['filename']
                    print(f"\n✅ MODÈLE MOBILENETV2 TROUVÉ: {model_to_load}")
                    print(f" → Utilisation de ce modèle par défaut (modèle léger et rapide)")
                    break
        
        if not model_to_load:
            for model_info in models_info:
                if 'vit' in model_info['filename'].lower() or model_info['filename'].endswith('.pth'):
                    model_to_load = model_info['filename']
                    print(f"\n✅ MODÈLE VISION TRANSFORMER TROUVÉ: {model_to_load}")
                    print(f" → Utilisation de ce modèle par défaut (architecture avancée)")
                    break
        
        if not model_to_load:
            for model_info in models_info:
                if 'resnet50' in model_info['filename'].lower():
                    model_to_load = model_info['filename']
                    print(f"\n✅ MODÈLE RESNET50 TROUVÉ: {model_to_load}")
                    print(f" → Utilisation de ce modèle par défaut")
                    break
        
        if not model_to_load and models_info:
            model_to_load = models_info[0]['filename']
            print(f"\n⚠️ Modèles prioritaires non trouvés")
            print(f" → Utilisation du modèle: {model_to_load}")
        
        if model_to_load:
            load_model(model_to_load)
        else:
            print(" ⚠️ AUCUN MODÈLE TROUVÉ DANS LE DOSSIER!")
    else:
        print(" ⚠️ AUCUN MODÈLE TROUVÉ!")
        print(f"\n" + "="*70)
        print("❌ ERREUR CRITIQUE: AUCUN MODÈLE DANS LE DOSSIER!")
        print("="*70)
        print("Veuillez placer dans le dossier models:")
        print("1. Modèles .h5 ou .keras (ResNet50, MobileNetV2, EfficientNetB0, ou autre CNN)")
        print("2. OU svm_model.pkl + scaler.pkl + pca.pkl + label_encoder.pkl (modèle SVM)")
        print("3. OU vit_best_model.pth (Vision Transformer)")
        return
    
    cam = Camera()
    
    print("\n" + "="*70)
    print("✅ APPLICATION PRÊTE")
    print("="*70)
    print(f"🌐 Interface web: http://localhost:5000/")
    print(f"📊 Modèle actif: {current_model_name or 'AUCUN'}")
    print(f"🎯 Type de modèle: {current_model_type or 'AUCUN'}")
    print(f"📷 Caméra: {'Active' if not cam.simulation else 'Mode Simulation'}")
    print(f"🎭 Plages d'âge: 1-20 (Jeunes), 21-50 (Adultes), 51-100 (Seniors)")
    print("="*70 + "\n")

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == '__main__':
    initialize_app()
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt")
    finally:
        if cam:
            cam.release()