import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# --- CONFIGURATION ---
MODEL_PATH = "models/model_raf.h5"
CLASS_NAMES = ['Surprise', 'Peur', 'Dégoût', 'Joie', 'Tristesse', 'Colère', 'Neutre']

# Stockage global du modèle et du détecteur de visages
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et le détecteur au démarrage, libère à l'arrêt."""
    print("🚀 Chargement du modèle...")
    state["model"] = load_model(MODEL_PATH)
    state["face_cascade"] = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print("✅ Modèle chargé.")
    yield
    state.clear()


app = FastAPI(title="RAF-DB Emotion Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
def test():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Lire les bytes de l'image
    contents = await file.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Image invalide ou format non supporté")

    # 2. Détection de visages
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = state["face_cascade"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return {"faces_detected": 0, "predictions": []}

    # 3. Construire un batch avec tous les visages détectés
    #    Un seul appel model.predict() pour tous les visages (plus efficace)
    batch = []
    boxes = []

    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (100, 100))
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        batch.append(img_rgb)
        boxes.append((x, y, w, h))

    # Normalisation VGG16 sur le batch complet : (N, 100, 100, 3)
    batch_preprocessed = preprocess_input(np.array(batch, dtype='float32'))

    # 4. Inférence en un seul appel pour tous les visages
    predictions = state["model"].predict(batch_preprocessed, verbose=0)

    # 5. Construire la liste des résultats
    results = []
    for i, (x, y, w, h) in enumerate(boxes):
        res_idx    = int(np.argmax(predictions[i]))
        confidence = float(np.max(predictions[i]))
        results.append({
            "emotion":    CLASS_NAMES[res_idx],
            "confidence": round(confidence * 100, 2),
            "box":        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        })

    return {"faces_detected": len(results), "predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=5000, reload=False)
