"""
test.py — Vérification de la configuration GPU / TensorFlow
============================================================
Lance ce script pour confirmer que TensorFlow détecte bien le GPU NVIDIA
et que les calculs s'exécutent correctement dessus.

Utilisation :  python test.py
"""

import tensorflow as tf


def testcarte():
    """
    Vérifie la présence d'un GPU NVIDIA et effectue un calcul de test.

    - Liste les GPUs physiques détectés par TensorFlow.
    - Affiche le nom du périphérique pour chaque GPU.
    - Effectue une multiplication de matrices 1000×1000 sur le GPU:0
      pour valider que le calcul est bien délégué au matériel graphique.
    """
    print("=" * 60)
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"GPUs détectés par TensorFlow : {[gpu.name for gpu in gpus]}")

        # Afficher le nom commercial de chaque GPU
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get("device_name", gpu.name)
            print(f"  -> {name}")

        # Test de calcul : multiplication de matrices aléatoires sur GPU:0
        print("\nTest de calcul sur GPU...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print("Multiplication de matrices 1000×1000 réussie sur GPU.")

    else:
        print("Aucun GPU NVIDIA détecté par TensorFlow.")
        print("L'entraînement s'exécutera sur CPU (beaucoup plus lent).")

    print("=" * 60)


if __name__ == "__main__":
    testcarte()
