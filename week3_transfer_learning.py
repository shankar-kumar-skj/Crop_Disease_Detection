"""
Week 3: Transfer Learning + Hyperparameter Optimization
Implements ResNet50 and MobileNetV2 transfer learning with fine-tuning.
Run after week2_cnn_baseline.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB3

# ─── Configuration ────────────────────────────────────────────────────────────
SPLIT_DIR = Path("data/split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
WARMUP_EPOCHS = 10      # train head only (base frozen)
FINETUNE_EPOCHS = 25    # unfreeze top layers and fine-tune

with open("dataset_metadata.json") as f:
    META = json.load(f)
NUM_CLASSES = META["num_classes"]


# ─── 1. Data generators (stronger augmentation for TL) ────────────────────────
def build_generators(img_size=IMG_SIZE):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.20,
        height_shift_range=0.20,
        horizontal_flip=True,
        zoom_range=0.25,
        brightness_range=[0.6, 1.4],
        channel_shift_range=20.0,
        shear_range=0.15,
        fill_mode="reflect"
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_ds = train_gen.flow_from_directory(
        SPLIT_DIR / "train", target_size=img_size,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True, seed=42)
    val_ds = val_gen.flow_from_directory(
        SPLIT_DIR / "val", target_size=img_size,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
    test_ds = val_gen.flow_from_directory(
        SPLIT_DIR / "test", target_size=img_size,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
    return train_ds, val_ds, test_ds


# ─── 2. Transfer learning model builder ───────────────────────────────────────
def build_transfer_model(backbone_name: str = "resnet50", num_classes: int = 38):
    """
    Strategy:
      Phase 1 - Frozen base: only train the new classification head.
      Phase 2 - Fine-tune:   unfreeze top N layers of base for domain adaptation.
    """
    input_tensor = layers.Input(shape=(*IMG_SIZE, 3))

    # Choose backbone
    backbone_map = {
        "resnet50":      (ResNet50,     {"include_top": False, "weights": "imagenet"}),
        "mobilenetv2":   (MobileNetV2,  {"include_top": False, "weights": "imagenet", "alpha": 1.0}),
        "efficientnetb3":(EfficientNetB3,{"include_top": False, "weights": "imagenet"}),
    }
    assert backbone_name in backbone_map, f"Unknown backbone: {backbone_name}"
    BackboneCls, kwargs = backbone_map[backbone_name]
    base = BackboneCls(input_tensor=input_tensor, **kwargs)
    base.trainable = False   # Phase 1: freeze entire base

    # Classification head
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=input_tensor, outputs=out,
                          name=f"TL_{backbone_name}")
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )


def get_callbacks(name):
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                    patience=3, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=f"models/{name}_best.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.CSVLogger(f"logs/{name}_history.csv"),
    ]


# ─── 3. Two-phase training ────────────────────────────────────────────────────
def two_phase_train(backbone_name: str, train_ds, val_ds):
    print(f"\n{'='*60}")
    print(f"  Transfer Learning — {backbone_name.upper()}")
    print(f"{'='*60}")

    model, base = build_transfer_model(backbone_name, NUM_CLASSES)
    name = backbone_name.lower()

    # ── Phase 1: Warmup — train head only ─────────────────────────────────────
    print(f"\n  Phase 1: Warmup ({WARMUP_EPOCHS} epochs, base frozen)")
    compile_model(model, lr=1e-3)
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=get_callbacks(f"{name}_phase1")
    )

    # ── Phase 2: Fine-tuning — unfreeze top layers ────────────────────────────
    print(f"\n  Phase 2: Fine-tune ({FINETUNE_EPOCHS} epochs, top layers unfrozen)")
    # Unfreeze last 30% of base layers
    unfreeze_from = int(len(base.layers) * 0.70)
    for layer in base.layers[unfreeze_from:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"  Trainable layers: {trainable_count}/{len(model.layers)}")

    compile_model(model, lr=1e-4)   # lower LR for fine-tuning
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=get_callbacks(f"{name}_phase2")
    )

    # Combine histories for plotting
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]

    return model, combined


# ─── 4. Learning rate sensitivity experiment ──────────────────────────────────
def lr_sensitivity_test(train_ds, val_ds, lrs=(1e-2, 1e-3, 1e-4)):
    """Quick 5-epoch test of different learning rates to find optimal."""
    print("\n  LR Sensitivity Test (5 epochs each)...")
    results = {}
    for lr in lrs:
        model, base = build_transfer_model("mobilenetv2", NUM_CLASSES)
        compile_model(model, lr)
        h = model.fit(train_ds, validation_data=val_ds,
                      epochs=5, verbose=0)
        final_val_acc = h.history["val_accuracy"][-1]
        results[lr] = final_val_acc
        print(f"    LR={lr:.0e}  →  val_acc={final_val_acc*100:.2f}%")
        del model
        tf.keras.backend.clear_session()
    return results


# ─── 5. Plot combined training curves ─────────────────────────────────────────
def plot_combined_history(history_dict: dict, name: str, warmup_ep: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("recall", "Recall")]

    for ax, (metric, label) in zip(axes, metrics):
        train_vals = history_dict[metric]
        val_vals = history_dict[f"val_{metric}"]
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, label="Train", linewidth=2)
        ax.plot(epochs, val_vals, label="Val", linewidth=2, linestyle="--")
        ax.axvline(warmup_ep, color="orange", linestyle=":", linewidth=1.5,
                   label="Fine-tune starts")
        ax.set_title(f"{name} — {label}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(f"plots/{name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_generators()

    # Optional: run LR sensitivity (comment out if you want to skip)
    lr_sensitivity_test(train_ds, val_ds)

    # Train ResNet50 (best accuracy, larger model)
    resnet_model, resnet_hist = two_phase_train("resnet50", train_ds, val_ds)
    plot_combined_history(resnet_hist, "ResNet50", WARMUP_EPOCHS)

    print("\n  Evaluating ResNet50 on test set...")
    res = resnet_model.evaluate(test_ds, verbose=1)
    print(f"  Test Accuracy:  {res[1]*100:.2f}%")
    print(f"  Test Recall:    {res[3]*100:.2f}%")

    # Train MobileNetV2 (faster, mobile-friendly)
    mobile_model, mobile_hist = two_phase_train("mobilenetv2", train_ds, val_ds)
    plot_combined_history(mobile_hist, "MobileNetV2", WARMUP_EPOCHS)

    print("\n  Evaluating MobileNetV2 on test set...")
    res2 = mobile_model.evaluate(test_ds, verbose=1)
    print(f"  Test Accuracy:  {res2[1]*100:.2f}%")
    print(f"  Test Recall:    {res2[3]*100:.2f}%")

    # Save the best model (choose whichever performed better)
    resnet_model.save("models/final_model_resnet50.keras")
    mobile_model.save("models/final_model_mobilenetv2.keras")
    print("\n  Week 3 complete. Run week4_evaluation_inference.py next.")
