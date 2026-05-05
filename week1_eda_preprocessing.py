import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path("plantvillage dataset/color")   # adjust if unzip path differs
OUTPUT_DIR = Path("data/processed")
SPLIT_DIR = Path("data/split")
IMG_SIZE = (224, 224)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── 1. Discover classes ──────────────────────────────────────────────────────
def discover_classes(data_dir: Path):
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"\n{'='*60}")
    print(f"  PlantVillage Dataset — {len(classes)} classes found")
    print(f"{'='*60}")
    for i, c in enumerate(classes):
        count = len(list((data_dir / c).glob("*.jpg"))) + \
                len(list((data_dir / c).glob("*.JPG"))) + \
                len(list((data_dir / c).glob("*.png")))
        print(f"  [{i+1:02d}] {c:<50} {count:>5} images")
    return classes


# ─── 2. EDA — class distribution plot ─────────────────────────────────────────
def plot_class_distribution(data_dir: Path, classes: list):
    counts = []
    for c in classes:
        imgs = list((data_dir / c).glob("*.[jJpP][pPnN][gG]"))
        counts.append(len(imgs))

    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ["#2ecc71" if "healthy" in c.lower() else "#e74c3c" for c in classes]
    bars = ax.barh(classes, counts, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Number of images", fontsize=12)
    ax.set_title("PlantVillage — Class Distribution", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend = [Patch(color="#2ecc71", label="Healthy"),
              Patch(color="#e74c3c", label="Diseased")]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Total images: {sum(counts):,}")
    print(f"  Min class: {min(counts)} | Max class: {max(counts)}")
    return counts


# ─── 3. EDA — sample image grid ───────────────────────────────────────────────
def plot_sample_images(data_dir: Path, classes: list, n_samples=5):
    sample_classes = random.sample(classes, min(8, len(classes)))
    fig, axes = plt.subplots(len(sample_classes), n_samples,
                              figsize=(n_samples * 3, len(sample_classes) * 3))
    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)

    for row, cls in enumerate(sample_classes):
        imgs = list((data_dir / cls).glob("*.[jJpP][pPnN][gG]"))[:n_samples]
        for col, img_path in enumerate(imgs):
            ax = axes[row][col]
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                label = cls.replace("___", "\n").replace("_", " ")
                ax.set_ylabel(label, fontsize=7, rotation=0,
                              labelpad=80, va="center")

    plt.tight_layout()
    plt.savefig("eda_sample_images.png", dpi=120, bbox_inches="tight")
    plt.show()


# ─── 4. EDA — pixel statistics ────────────────────────────────────────────────
def analyze_pixel_stats(data_dir: Path, classes: list, sample_per_class=30):
    print("\n  Computing pixel mean/std across dataset (sampled)...")
    means, stds, widths, heights = [], [], [], []

    for cls in classes:
        imgs = list((data_dir / cls).glob("*.[jJpP][pPnN][gG]"))
        sample = random.sample(imgs, min(sample_per_class, len(imgs)))
        for p in sample:
            img = np.array(Image.open(p).convert("RGB")) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)

    global_mean = np.array(means).mean(axis=0)
    global_std = np.array(stds).mean(axis=0)
    print(f"  Global mean (RGB): {global_mean.round(4)}")
    print(f"  Global std  (RGB): {global_std.round(4)}")
    print(f"  Image sizes — W: min={min(widths)} max={max(widths)} | "
          f"H: min={min(heights)} max={max(heights)}")
    return global_mean, global_std


# ─── 5. Train / Val / Test split ──────────────────────────────────────────────
def create_splits(data_dir: Path, split_dir: Path, classes: list):
    print(f"\n  Creating train/val/test splits → {split_dir}")
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        (split_dir / split).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        imgs = list((data_dir / cls).glob("*.[jJpP][pPnN][gG]"))
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:]
        }

        for split, files in splits.items():
            dest = split_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
            split_counts[split] += len(files)

    print(f"  Train: {split_counts['train']:,} | "
          f"Val: {split_counts['val']:,} | "
          f"Test: {split_counts['test']:,}")

    # Save metadata
    meta = {"classes": classes, "num_classes": len(classes),
            "split_counts": split_counts, "img_size": list(IMG_SIZE)}
    with open("dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved dataset_metadata.json")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    classes = discover_classes(DATA_DIR)
    counts = plot_class_distribution(DATA_DIR, classes)
    plot_sample_images(DATA_DIR, classes)
    global_mean, global_std = analyze_pixel_stats(DATA_DIR, classes)
    create_splits(DATA_DIR, SPLIT_DIR, classes)
    print("\n  Week 1 complete. Run week2_cnn_baseline.py next.")
