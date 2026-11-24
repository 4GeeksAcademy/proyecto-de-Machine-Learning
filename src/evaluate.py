"""Evalua models/best_model.keras con el conjunto de test y guarda el modelo final."""
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ts = ImageDataGenerator().flow_from_directory(
    f"{ROOT}/data/processed/test", target_size=(200, 200), batch_size=32,
    class_mode="categorical", shuffle=False)
best = load_model(f"{ROOT}/models/best_model.keras")
pred = np.argmax(best.predict(ts, verbose=0), axis=1)
print("Accuracy test:", (pred == ts.classes).mean())
print(confusion_matrix(ts.classes, pred))
print(classification_report(ts.classes, pred, target_names=list(ts.class_indices)))
best.save(f"{ROOT}/models/cats_dogs_model.keras")
print("Modelo guardado en models/cats_dogs_model.keras")
