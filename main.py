# ============================================
# BRAIN TUMOR CLASSIFIER - CNN
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ── STEP 1: SETTINGS ────────────────────────

TRAIN_DIR  = "Training"
TEST_DIR   = "Testing"
IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 20

# ── STEP 2: LOAD DATA ───────────────────────

# Augment training images to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # normalize pixel values 0–1
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2      # 20% of training data → validation
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)
CLASS_NAMES = list(train_data.class_indices.keys())
print("Classes:", CLASS_NAMES)

# ── STEP 3: BUILD MODEL ─────────────────────

model = models.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Flatten + Fully Connected
    layers.Flatten(),
    layers.Dropout(0.5),               # reduces overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")  # output layer

])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── STEP 4: TRAIN ───────────────────────────

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ── STEP 5: TEST ────────────────────────────

loss, accuracy = model.evaluate(test_data)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ── STEP 6: PLOT RESULTS ────────────────────

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],     label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],     label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ── STEP 7: SAVE MODEL ──────────────────────

model.save("brain_tumor_classifier_v3.keras")
print("Model saved!")

# ── STEP 8: PREDICT ONE IMAGE ───────────────

def predict_image(image_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img   = load_img(image_path, target_size=IMG_SIZE)
    arr   = img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    label = CLASS_NAMES[np.argmax(preds)]
    conf  = np.max(preds) * 100

    print(f"Prediction : {label}  ({conf:.1f}% confidence)")

    plt.imshow(img)
    plt.title(f"{label} ({conf:.1f}%)")
    plt.axis("off")
    plt.show()

