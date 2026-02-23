"""
app.py — API Flask pour la reconnaissance d'émotions faciales (RAF-DB)
======================================================================
Charge le modèle VGG16 fine-tuné une seule fois au démarrage, puis expose
un endpoint POST /predict qui :
  1. Reçoit une image multipart/form-data (champ 'file')
  2. Détecte les visages avec le Haar Cascade d'OpenCV
  3. Prétraite chaque visage (100×100, normalisation VGG16)
  4. Retourne l'émotion prédite et le niveau de confiance pour chaque visage

Lancement :  python app.py  →  http://0.0.0.0:5000
"""

import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask_cors import CORS

# ── Configuration ──────────────────────────────────────────────────────────────

# Chemin vers le modèle sauvegardé après entraînement
MODEL_PATH = "models/model_raf.h5"

# Classes dans l'ordre des indices du dataset RAF-DB (labels 1→7 mappés 0→6)
CLASS_NAMES = ['Surprise', 'Peur', 'Dégoût', 'Joie', 'Tristesse', 'Colère', 'Neutre']

# ── Initialisation de l'application ────────────────────────────────────────────

app = Flask(__name__)

# Autorise les requêtes cross-origin (nécessaire pour index.html ouvert en file://)
CORS(app)

# Chargement du modèle et du détecteur UNE SEULE FOIS au démarrage du serveur
# (évite de recharger 196 Mo à chaque requête)
print("Chargement du modèle en mémoire...")
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("Modèle chargé. Serveur prêt.")

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/test', methods=['GET'])
def test():
    """Endpoint de vérification : confirme que le serveur est opérationnel."""
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit l'émotion de chaque visage présent dans l'image envoyée.

    Entrée  : multipart/form-data avec un champ 'file' contenant l'image.
    Sortie  : JSON { "faces_detected": N, "predictions": [...] }
              Chaque prédiction contient : emotion, confidence (%), box (x,y,w,h).
    """

    # 1. Vérifier qu'une image est bien présente dans la requête
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files['file']

    # 2. Décoder les bytes de l'image en tableau OpenCV (BGR)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image invalide ou format non supporté"}), 400

    # 3. Détection des visages sur l'image en niveaux de gris
    #    scaleFactor=1.1 : agrandissement de 10 % à chaque échelle
    #    minNeighbors=5  : filtre les faux positifs (plus élevé = plus strict)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results = []

    # 4. Pour chaque visage détecté, prédire l'émotion
    for (x, y, w, h) in faces:

        # Découper la région d'intérêt (ROI) et la redimensionner en 100×100
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (100, 100))

        # Convertir BGR→RGB (OpenCV lit en BGR, mais VGG16 attend du RGB)
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Ajouter la dimension batch : (100,100,3) → (1,100,100,3)
        img_array = np.expand_dims(img_rgb, axis=0)

        # Normalisation propre à VGG16 : soustraction de la moyenne ImageNet
        img_preprocessed = preprocess_input(img_array.astype('float32'))

        # Inférence : vecteur de probabilités sur les 7 classes
        prediction = model.predict(img_preprocessed, verbose=0)
        res_idx   = int(np.argmax(prediction[0]))     # indice de la classe prédite
        confidence = float(np.max(prediction[0]))     # probabilité associée

        results.append({
            "emotion":    CLASS_NAMES[res_idx],
            "confidence": round(confidence * 100, 2),  # en pourcentage
            "box":        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        })

    # 5. Retourner la liste des prédictions (liste vide si aucun visage trouvé)
    return jsonify({
        "faces_detected": len(results),
        "predictions":    results
    })


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # host='0.0.0.0' rend le serveur accessible depuis l'extérieur du conteneur
    app.run(host='0.0.0.0', port=5000)
