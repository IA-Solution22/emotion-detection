# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Facial Emotion Recognition (FER) system trained on the RAF-DB dataset. Classifies 7 emotions from facial images using VGG16 transfer learning, served via a Flask REST API with a vanilla JS frontend.

## Running the Project

```bash
# Start the Flask inference API (loads models/model_raf.h5 on startup)
python app.py
# Runs on 0.0.0.0:5000

# Verify GPU/TensorFlow setup
python test/testgpu.py

# Open index.html directly in a browser (requires the Flask API to be running)
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
├── app.py                  # API Flask principale (inference)
├── app_fastapi.py          # API FastAPI alternative
├── index.html              # Frontend vanilla JS
├── CLAUDE.md
│
├── models/                 # Modèles entraînés
│   ├── model_raf.h5                  # Modèle principal (196 MB, chargé par app.py)
│   ├── VGG16_RAFDB_84_percent.keras  # Checkpoint alternatif
│   └── VGG16_RAFDB_percent.keras     # Checkpoint alternatif
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

### Inference Pipeline (`app.py`)
1. Au démarrage : charge `models/model_raf.h5` et le Haar Cascade OpenCV
2. `POST /predict` : décode l'image → détection visages en niveaux de gris (`detectMultiScale(gray, 1.1, 5)`) → pour chaque visage : crop, resize 100×100, BGR→RGB, `preprocess_input` VGG16, inférence, retourne classe + confiance
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
