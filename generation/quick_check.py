#!/usr/bin/env python3
"""
Quick Visual Check - View latest training samples
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def check_latest_samples():
    print("🔍 Checking latest training samples...")

    sample_dir = "image_samples"
    if not os.path.exists(sample_dir):
        print("❌ No image_samples directory found!")
        return

    sample_files = [f for f in os.listdir(sample_dir) if f.startswith("image_at_epoch_") and f.endswith(".png")]

    if not sample_files:
        print("❌ No sample images found!")
        return

    # Sort by epoch number
    sample_files.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))

    # Show the latest few samples
    latest_files = sample_files[-3:]  # Last 3 samples

    print(f"📊 Found {len(sample_files)} sample images")
    print(f"🎯 Showing latest samples: {[f.replace('.png', '') for f in latest_files]}")

    fig, axes = plt.subplots(1, len(latest_files), figsize=(5 * len(latest_files), 5))

    if len(latest_files) == 1:
        axes = [axes]

    for i, filename in enumerate(latest_files):
        filepath = os.path.join(sample_dir, filename)
        img = mpimg.imread(filepath)

        axes[i].imshow(img)
        epoch_num = filename.split("_")[3].split(".")[0]
        axes[i].set_title(f"Epoch {epoch_num}", fontsize=14)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Show progression
    print("\n📈 Training Progression:")
    for filename in sample_files[-5:]:  # Last 5 epochs
        epoch_num = filename.split("_")[3].split(".")[0]
        print(f"   ✓ Epoch {epoch_num}")


if __name__ == "__main__":
    check_latest_samples()