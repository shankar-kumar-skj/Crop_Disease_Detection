"""
Week 2: Custom CNN Architecture and Baseline Training
Builds a from-scratch CNN with Dropout + Early Stopping to establish baseline accuracy.
Run after week1_eda_preprocessing.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Configuration ────────────────────────────────────────────────────────────
SPLIT_DIR = Path("data/split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

with open("dataset_metadata.json") as f:
    META = json.load(f)
NUM_CLASSES = META["num_classes"]
CLASS_NAMES = META["classes"]

print(f"  Classes: {NUM_CLASSES} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")


# ─── 1. Data generators with augmentation ─────────────────────────────────────
def build_generators():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        fill_mode="nearest"
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_ds = train_gen.flow_from_directory(
        SPLIT_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    val_ds = val_gen.flow_from_directory(
        SPLIT_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    test_ds = val_gen.flow_from_directory(
        SPLIT_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return train_ds, val_ds, test_ds


# ─── 2. Custom CNN architecture ───────────────────────────────────────────────
def build_custom_cnn(num_classes: int) -> tf.keras.Model:
    """
    Architecture:
      Block 1: Conv2D(32)  → BN → ReLU → Conv2D(32)  → BN → ReLU → MaxPool → Dropout
      Block 2: Conv2D(64)  → BN → ReLU → Conv2D(64)  → BN → ReLU → MaxPool → Dropout
      Block 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout
      Block 4: Conv2D(256) → BN → ReLU → MaxPool → Dropout
      Head:    GlobalAvgPool → Dense(512) → BN → ReLU → Dropout → Dense(num_classes)
    """
    inp = layers.Input(shape=(*IMG_SIZE, 3))

    def conv_block(x, filters, dropout_rate=0.25):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    x = conv_block(inp, 32, 0.20)
    x = conv_block(x, 64, 0.25)
    x = conv_block(x, 128, 0.30)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.35)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.50)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="CustomCNN")
    return model


# ─── 3. Callbacks ─────────────────────────────────────────────────────────────
def build_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath="models/custom_cnn_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir="logs/custom_cnn", histogram_freq=1),
        callbacks.CSVLogger("logs/custom_cnn_history.csv")
    ]


# ─── 4. Training ──────────────────────────────────────────────────────────────
def train(model, train_ds, val_ds):
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    model.summary()

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=build_callbacks()
    )
    return history


# ─── 5. Plot training curves ──────────────────────────────────────────────────
def plot_history(history, title="Custom CNN"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("recall", "Recall")]

    for ax, (metric, label) in zip(axes, metrics):
        ax.plot(history.history[metric], label=f"Train {label}", linewidth=2)
        ax.plot(history.history[f"val_{metric}"], label=f"Val {label}",
                linewidth=2, linestyle="--")
        ax.set_title(f"{title} — {label}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}_curves.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Path("plots").mkdir(exist_ok=True)

    print("\n  Building data generators...")
    train_ds, val_ds, test_ds = build_generators()

    print("\n  Building Custom CNN...")
    model = build_custom_cnn(NUM_CLASSES)

    print("\n  Training...")
    history = train(model, train_ds, val_ds)

    print("\n  Evaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss:      {results[0]:.4f}")
    print(f"  Test Accuracy:  {results[1]*100:.2f}%")
    print(f"  Test Precision: {results[2]*100:.2f}%")
    print(f"  Test Recall:    {results[3]*100:.2f}%")

    plot_history(history, title="Custom CNN")
    print("\n  Week 2 complete. Run week3_transfer_learning.py next.")
