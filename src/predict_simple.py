import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ===============================
# 1. PARAMÈTRES
# ===============================
IMAGE_PATH = "data/test.png"
MODEL_PATH = "models/resnet50_advanced_final.h5"

CLASSES = ['1-20', '21-50', '51-100']
IMG_SIZE = (224, 224)

# ===============================
# 2. CHARGER LE MODÈLE
# ===============================
def categorical_crossentropy_with_label_smoothing(y_true, y_pred, smoothing=0.1):
    y_true = y_true * (1.0 - smoothing) + smoothing / tf.cast(tf.shape(y_pred)[-1], tf.float32)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "categorical_crossentropy_with_label_smoothing":
            categorical_crossentropy_with_label_smoothing
        }
    )
    print("✅ Modèle chargé avec succès")
except:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("⚠️ Modèle chargé sans custom_objects")

# ===============================
# 3. CHARGER ET PRÉPARER L'IMAGE
# ===============================
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("❌ Image non trouvée")

img = cv2.resize(img, IMG_SIZE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.expand_dims(img, axis=0)
img = preprocess_input(img.astype(np.float32))

# ===============================
# 4. PRÉDICTION
# ===============================
predictions = model.predict(img)[0]

pred_idx = np.argmax(predictions)
pred_class = CLASSES[pred_idx]
confidence = predictions[pred_idx]

# ===============================
# 5. AFFICHAGE DES RÉSULTATS
# ===============================
print("\n🎯 RÉSULTAT DE LA PRÉDICTION")
print(f"Classe prédite : {pred_class}")
print(f"Confiance      : {confidence:.2%}")

print("\n📊 Détails des probabilités :")
for i, cls in enumerate(CLASSES):
    print(f" - {cls} : {predictions[i]:.2%}")
