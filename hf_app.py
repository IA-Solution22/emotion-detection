import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Support HEIC/HEIF (photos iPhone)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MODEL_PATH = "models/model_raf.h5"

CLASS_NAMES = ['Surprise', 'Peur', 'Dégoût', 'Joie', 'Tristesse', 'Colère', 'Neutre']

EMOTION_EMOJI = {
    'Surprise':  '😮',
    'Peur':      '😨',
    'Dégoût':    '🤢',
    'Joie':      '😄',
    'Tristesse': '😢',
    'Colère':    '😠',
    'Neutre':    '😐',
}

CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
h1 {
    background: linear-gradient(90deg, #e94560, #f5a623, #f7e017, #56e0a0, #4fc3f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-align: center;
    padding-bottom: 0.5rem;
}
.st-key-chatbot {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 3px solid #4fc3f7 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 28px rgba(79, 195, 247, 0.5), 0 0 10px rgba(79, 195, 247, 0.2) !important;
    padding: 1.2rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(79, 195, 247, 0.15), rgba(86, 224, 160, 0.15)) !important;
    border-radius: 12px;
    border-left: 3px solid #4fc3f7;
    padding: 0.5rem;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(233, 69, 96, 0.15), rgba(245, 166, 35, 0.15)) !important;
    border-radius: 12px;
    border-left: 3px solid #e94560;
    padding: 0.5rem;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #e8eaf6 !important;
}
.stButton button {
    background: linear-gradient(135deg, #e94560, #f5a623) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
}
.stButton button:hover,
.stButton button:focus,
.stButton button:active {
    background: linear-gradient(135deg, #e94560, #f5a623) !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
}
p, li, label {
    color: #e8eaf6 !important;
}
</style>
"""


@st.cache_resource(show_spinner="Chargement du modèle IA... (première fois ~60s)")
def load_resources():
    import os
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.vgg16 import preprocess_input
    model = load_model(MODEL_PATH)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, cascade, preprocess_input


def image_bytes_to_cv2(image_bytes):
    from PIL import ImageOps
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img_rgb = np.array(pil_img)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def predict(image_bytes):
    try:
        model, cascade, preprocess_input = load_resources()
        img = image_bytes_to_cv2(image_bytes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return 0, [], None

        batch, boxes = [], []
        for (x, y, w, h) in faces:
            roi = cv2.resize(img[y:y + h, x:x + w], (100, 100))
            batch.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            boxes.append((x, y, w, h))

        preds = model.predict(
            preprocess_input(np.array(batch, dtype="float32")), verbose=0
        )

        results = []
        for i in range(len(boxes)):
            idx = int(np.argmax(preds[i]))
            results.append({
                "emotion":    CLASS_NAMES[idx],
                "confidence": round(float(np.max(preds[i])) * 100, 2),
            })

        return len(results), results, None

    except Exception as e:
        return 0, [], str(e)


def run_prediction(image_bytes):
    with st.spinner("Analyse en cours..."):
        faces, predictions, error = predict(image_bytes)

    if error:
        st.session_state.result_text = f"❌ Erreur : {error}"
    elif faces == 0:
        st.session_state.result_text = "😕 Aucun visage détecté dans la photo."
    elif faces == 1:
        pred = predictions[0]
        emoji = EMOTION_EMOJI.get(pred["emotion"], "")
        st.session_state.result_text = f"{emoji} {pred['emotion']} — {pred['confidence']:.1f}% de confiance"
    else:
        lines = []
        for i, pred in enumerate(predictions, 1):
            emoji = EMOTION_EMOJI.get(pred["emotion"], "")
            lines.append(f"{i}. {emoji} {pred['emotion']} — {pred['confidence']:.1f}% de confiance")
        st.session_state.result_text = "\n".join(lines)

    st.session_state.result_image = image_bytes
    st.rerun()


def detect_mobile():
    try:
        ua = st.context.headers.get("User-Agent", "")
        return any(x in ua.lower() for x in ["mobile", "android", "iphone", "ipad"])
    except Exception:
        return False


# --- INIT ---
st.set_page_config(page_title="Détection d'émotions", page_icon="😊")
st.markdown(CSS, unsafe_allow_html=True)
st.title("Détection d'émotions faciales")

try:
    load_resources()
except Exception as e:
    st.error(f"Erreur au chargement du modèle : {e}")
    st.stop()

is_mobile = detect_mobile()

if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "mode" not in st.session_state:
    st.session_state.mode = None
if "result_text" not in st.session_state:
    st.session_state.result_text = None
if "result_image" not in st.session_state:
    st.session_state.result_image = None

with st.container(border=True, key="chatbot"):

    # =========================================================
    # VERSION MOBILE — caméra uniquement
    # =========================================================
    if is_mobile:
        with st.chat_message("assistant"):
            st.write("""Bonjour ! 👋 Je suis un assistant de **reconnaissance d'émotions faciales**.

Je peux analyser une photo de visage et détecter parmi 7 émotions :

😮 Surprise · 😨 Peur · 🤢 Dégoût · 😄 Joie · 😢 Tristesse · 😠 Colère · 😐 Neutre

📷 Prenez une photo pour commencer.""")

        if st.session_state.result_image is not None:
            with st.chat_message("user"):
                st.image(st.session_state.result_image, width=300)
            with st.chat_message("assistant"):
                st.write(st.session_state.result_text)
            if st.button("📷 Nouvelle photo", use_container_width=True):
                st.session_state.result_text = None
                st.session_state.result_image = None
                st.session_state.input_key += 1
                st.rerun()
        else:
            photo = st.camera_input("Caméra", key=f"camera_{st.session_state.input_key}")
            if photo:
                run_prediction(photo.getvalue())

    # =========================================================
    # VERSION DESKTOP — caméra + upload fichier
    # =========================================================
    else:
        with st.chat_message("assistant"):
            st.write("""Bonjour ! 👋 Je suis un assistant de **reconnaissance d'émotions faciales**.

Je peux analyser une photo de visage et détecter parmi 7 émotions :

😮 Surprise · 😨 Peur · 🤢 Dégoût · 😄 Joie · 😢 Tristesse · 😠 Colère · 😐 Neutre

Comment souhaitez-vous procéder ?

**1️⃣ Utiliser la caméra** — prenez une photo en direct
**2️⃣ Charger un fichier** — sélectionnez une image depuis votre PC""")

        # Choix du mode
        if st.session_state.mode is None:
            st.session_state.result_text = None
            st.session_state.result_image = None
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📷 1 — Utiliser la caméra", use_container_width=True):
                    st.session_state.mode = "camera"
                    st.rerun()
            with col2:
                if st.button("📁 2 — Charger un fichier", use_container_width=True):
                    st.session_state.mode = "file"
                    st.rerun()

        else:
            # Résultat commun aux deux modes desktop
            if st.session_state.result_image is not None:
                with st.chat_message("user"):
                    st.image(st.session_state.result_image, width=300)
                with st.chat_message("assistant"):
                    st.write(st.session_state.result_text)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Continuer", use_container_width=True):
                        st.session_state.result_text = None
                        st.session_state.result_image = None
                        st.session_state.input_key += 1
                        st.rerun()
                with col2:
                    if st.button("↩️ Changer de mode", use_container_width=True):
                        st.session_state.mode = None
                        st.session_state.result_text = None
                        st.session_state.result_image = None
                        st.session_state.input_key += 1
                        st.rerun()

            # --- MODE CAMÉRA desktop ---
            elif st.session_state.mode == "camera":
                with st.chat_message("assistant"):
                    st.write("📷 Prenez une photo avec votre caméra :")
                photo = st.camera_input("Caméra", key=f"camera_{st.session_state.input_key}")
                if photo:
                    run_prediction(photo.getvalue())

            # --- MODE FICHIER desktop ---
            elif st.session_state.mode == "file":
                with st.chat_message("assistant"):
                    st.write("📁 Sélectionnez une image depuis votre PC :")
                uploaded = st.file_uploader(
                    "Image",
                    type=["jpg", "jpeg", "png", "webp", "heic", "heif", "bmp"],
                    label_visibility="collapsed",
                    key=f"file_{st.session_state.input_key}"
                )
                if uploaded is not None:
                    run_prediction(uploaded.getvalue())
