# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Facial Emotion Recognition (FER) system trained on the RAF-DB dataset. Classifies 7 emotions from facial images using VGG16 transfer learning, served via a FastAPI REST API with a vanilla JS frontend.

## Running the Project

```bash
# Start the FastAPI inference API (loads models/model_raf.h5 on startup)
python app_fastapi.py
# Runs on 0.0.0.0:5000

# Alternative : API Flask
python app.py
# Runs on 0.0.0.0:5000

# Verify GPU/TensorFlow setup
python test/testgpu.py

# Open index.html directly in a browser (requires the API to be running)

# Streamlit chatbot with camera (requires the API to be running)
pip install streamlit requests
streamlit run streamlit_app.py
```

The web UI (`index.html`) calls `http://localhost:5000/predict` via `fetch`. The API accepts a `multipart/form-data` POST with a `file` field and returns JSON.

## Environment

The devcontainer uses `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` with `--gpus all`. Key version constraints from `postCreateCommand`:
- `numpy<2.0.0` (confirmed 1.26.4)
- `scipy<1.13.0`
- `opencv-python-headless<4.10`

The DATASET directory is bind-mounted from the host (`c:/ProjetIA/raf-db/DATASET` → `/workspace/DATASET`).

## Project Structure

```
raf-db/
├── app_fastapi.py          # API FastAPI principale (inference) — à utiliser
├── app.py                  # API Flask alternative
├── index.html              # Frontend vanilla JS
├── CLAUDE.md
│
├── css/                    # Feuilles de style
│   └── style.css           # Styles du frontend (extrait de index.html)
│
├── models/                 # Modèles entraînés
│   └── model_raf.h5        # Modèle principal (196 MB, suivi via Git LFS)
│
├── notebooks/              # Notebooks d'entraînement
│   ├── raf transfert VGG16.ipynb     # Principal — VGG16 fine-tuning (meilleurs résultats)
│   ├── raf transfert.ipynb           # Autres expériences de transfer learning
│   ├── raf simple.ipynb              # Baseline CNN from scratch
│   ├── raf augmentation.ipynb        # Exploration de l'augmentation de données
│   └── raf poid.ipynb                # Expériences de pondération des classes
│
├── results/                # Graphiques et matrices générés lors de l'entraînement
│   ├── accuracy.png
│   ├── distribution_emotions.png
│   ├── matricec.png
│   ├── model.png
│   ├── output accuracy augm.png
│   └── outputcurrancy poid.png
│
├── samples/                # Images de test manuelles (démo)
│
├── test/                   # Scripts de vérification
│   └── testgpu.py          # Vérifie la détection GPU par TensorFlow
│
├── doc/                    # Documentation du projet
│   ├── presentation.docx
│   └── 2026-01-28 19-07-03.mp4
│
└── DATASET/                # Dataset RAF-DB (bind-mount depuis l'hôte)
    ├── train/              # Sous-dossiers 1–7 (label émotion), 12 271 images
    ├── test/               # Sous-dossiers 1–7, 3 068 images
    ├── train_labels.csv
    └── test_labels.csv
```

## Architecture

### Inference Pipeline (`app_fastapi.py`)
1. Au démarrage (lifespan) : charge `models/model_raf.h5` et le Haar Cascade OpenCV
2. `POST /predict` : décode l'image → détection visages en niveaux de gris (`detectMultiScale(gray, 1.1, 5)`) → regroupe tous les visages en un batch → un seul appel `model.predict()` → retourne classe + confiance pour chaque visage
3. Réponse : `{ "faces_detected": N, "predictions": [{ "emotion", "confidence", "box" }] }`

### Emotion Classes
Indexed 0–6, mapped from RAF-DB labels 1–7:
`['Surprise', 'Peur', 'Dégoût', 'Joie', 'Tristesse', 'Colère', 'Neutre']`

Class imbalance is significant: Joie has 1,185 test samples vs. Peur with 74.

### Model Architecture (from `notebooks/raf transfert VGG16.ipynb`)
- VGG16 base (ImageNet weights, all layers unfrozen for fine-tuning)
- Custom head: Flatten → Dense(512, relu) + BatchNorm → Dropout(0.5) → Dense(7, softmax)
- Input: 100×100 RGB images
- Training: Adam(lr=1e-4), sparse_categorical_crossentropy, EarlyStopping(patience=8), ReduceLROnPlateau
- Achieved ~82.4% weighted accuracy; Joie F1=0.93, Dégoût F1=0.41 (minority class)

Training notebooks use `ImageDataGenerator.flow_from_dataframe` with 18 workers and multiprocessing.
