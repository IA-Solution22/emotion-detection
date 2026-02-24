import streamlit as st
import requests

API_URL = "http://localhost:5000/predict"

EMOTION_EMOJI = {
    'Surprise':  '😮',
    'Peur':      '😨',
    'Dégoût':    '🤢',
    'Joie':      '😄',
    'Tristesse': '😢',
    'Colère':    '😠',
    'Neutre':    '😐',
}

st.set_page_config(page_title="Détection d'émotions", page_icon="😊")
st.title("Détection d'émotions faciales")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

photo = st.camera_input("Prenez une photo pour analyser votre émotion")

if photo:
    with st.chat_message("user"):
        st.image(photo, width=300)
    st.session_state.messages.append({"role": "user", "content": "📷 Photo envoyée"})

    try:
        response = requests.post(
            API_URL,
            files={"file": ("photo.jpg", photo.getvalue(), "image/jpeg")},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        faces = data.get("faces_detected", 0)
        predictions = data.get("predictions", [])

        if faces == 0:
            bot_text = "😕 Aucun visage détecté dans la photo."
        elif faces == 1:
            pred = predictions[0]
            emotion = pred["emotion"]
            confidence = pred["confidence"] * 100
            emoji = EMOTION_EMOJI.get(emotion, "")
            bot_text = f"{emoji} {emotion} — {confidence:.1f}% de confiance"
        else:
            lines = []
            for i, pred in enumerate(predictions, 1):
                emotion = pred["emotion"]
                confidence = pred["confidence"] * 100
                emoji = EMOTION_EMOJI.get(emotion, "")
                lines.append(f"{i}. {emoji} {emotion} — {confidence:.1f}% de confiance")
            bot_text = "\n".join(lines)

    except requests.exceptions.ConnectionError:
        bot_text = "Impossible de contacter l'API. Assurez-vous que le serveur est démarré : `python app_fastapi.py`"
    except requests.exceptions.Timeout:
        bot_text = "L'API n'a pas répondu dans les 10 secondes. Vérifiez que le serveur est opérationnel."
    except Exception as e:
        bot_text = f"Erreur inattendue : {e}"

    with st.chat_message("assistant"):
        st.write(bot_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_text})
