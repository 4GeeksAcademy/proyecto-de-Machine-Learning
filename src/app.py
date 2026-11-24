"""Clasificador perros vs gatos con EfficientNetB0 (transfer learning).

Uso: python src/app.py   (desde la raiz del proyecto)
Requiere data/processed/{train,test}/{cats,dogs} (se genera en explore.ipynb).
"""
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
SIZE, BATCH, EPOCHS = (200, 200), 32, 3

# EfficientNet de Keras ya incluye su normalizacion: NO usar rescale=1/255.
trdata = ImageDataGenerator().flow_from_directory(
    f"{DATA}/train", target_size=SIZE, batch_size=BATCH, class_mode="categorical")
# validacion mezclada (si no, los primeros lotes serian de una sola clase)
valdata = ImageDataGenerator().flow_from_directory(
    f"{DATA}/test", target_size=SIZE, batch_size=BATCH, class_mode="categorical", shuffle=True, seed=1)
# test ordenado, para metricas finales
tsdata = ImageDataGenerator().flow_from_directory(
    f"{DATA}/test", target_size=SIZE, batch_size=BATCH, class_mode="categorical", shuffle=False)

base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=SIZE + (3,))
base.trainable = False  # CPU: solo entrenamos la cabeza

model = Sequential([base, GlobalAveragePooling2D(),
                    Dense(128, activation="relu"), Dense(2, activation="softmax")])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

best_path = os.path.join(MODELS, "best_model.keras")
callbacks = [
    ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
]
model.fit(trdata, validation_data=valdata, epochs=EPOCHS, callbacks=callbacks)

best = load_model(best_path)
probs = best.predict(tsdata)
pred = np.argmax(probs, axis=1)
print("Accuracy test:", (pred == tsdata.classes).mean())
print(confusion_matrix(tsdata.classes, pred))
print(classification_report(tsdata.classes, pred, target_names=list(tsdata.class_indices)))

best.save(os.path.join(MODELS, "cats_dogs_model.keras"))
print("Modelo guardado en models/cats_dogs_model.keras")
