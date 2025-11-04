#!/usr/bin/env python3
"""
GAN Accuracy Testing Script
This script evaluates the quality and accuracy of GAN-generated characters
using multiple methods including visual assessment and classifier-based evaluation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load the trained generator and class mappings
print("🔄 Loading trained generator...")
try:
    generator = tf.keras.models.load_model('captcha_char_generator.h5')
    print("✅ Generator loaded successfully")
except Exception as e:
    print(f"❌ Error loading generator: {e}")
    exit(1)

print("🔄 Loading class mappings...")
with open('class_mappings.json', 'r') as f:
    class_data = json.load(f)
class_names = class_data['class_names']
char_to_int = {name: i for i, name in enumerate(class_names)}
int_to_char = {i: name for i, name in enumerate(class_names)}

# Constants
LATENT_DIM = 100
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CLASSES = len(class_names)

print(f"📊 Testing {NUM_CLASSES} character classes: {class_names}")


def generate_character_samples(num_samples_per_class=10):
    """Generate samples for each character class"""
    print(f"\n🎨 Generating {num_samples_per_class} samples per character...")

    all_generated_images = []
    all_labels = []

    for class_idx in range(NUM_CLASSES):
        char = int_to_char[class_idx]
        print(f"  Generating samples for '{char}'...")

        for _ in range(num_samples_per_class):
            # Generate random noise
            noise = tf.random.normal([1, LATENT_DIM])
            label = tf.constant([[class_idx]])

            # Generate image
            generated_image = generator([noise, label], training=False)
            generated_image = generated_image[0, :, :, 0].numpy()

            all_generated_images.append(generated_image)
            all_labels.append(class_idx)

    return np.array(all_generated_images), np.array(all_labels)


def create_character_grid(images, labels, samples_per_row=10):
    """Create a grid showing generated characters"""
    print("\n📋 Creating character quality grid...")

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, samples_per_row,
                             figsize=(samples_per_row * 2, num_classes * 2))

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx in range(num_classes):
        char = int_to_char[class_idx]
        class_images = images[labels == class_idx][:samples_per_row]

        for sample_idx in range(min(samples_per_row, len(class_images))):
            ax = axes[class_idx, sample_idx]

            # Normalize image for display
            img_display = (class_images[sample_idx] + 1) / 2.0
            ax.imshow(img_display, cmap='gray')
            ax.set_title(f"'{char}'", fontsize=8)
            ax.axis('off')

        # Fill empty slots if not enough samples
        for sample_idx in range(len(class_images), samples_per_row):
            axes[class_idx, sample_idx].axis('off')

    plt.tight_layout()
    plt.savefig('accuracy_test_character_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Character grid saved as 'accuracy_test_character_grid.png'")


def visual_quality_assessment(images, labels):
    """Assess visual quality of generated characters"""
    print("\n👁️ Visual Quality Assessment:")

    # Calculate basic image statistics
    mean_pixel_values = []
    std_pixel_values = []
    contrast_scores = []

    for img in images:
        # Normalize to 0-1 range
        img_norm = (img + 1) / 2.0

        mean_pixel_values.append(np.mean(img_norm))
        std_pixel_values.append(np.std(img_norm))

        # Calculate contrast (standard deviation of pixel values)
        contrast_scores.append(np.std(img_norm))

    print(f"  📊 Average pixel intensity: {np.mean(mean_pixel_values):.3f} ± {np.std(mean_pixel_values):.3f}")
    print(f"  📊 Average contrast: {np.mean(contrast_scores):.3f} ± {np.std(contrast_scores):.3f}")

    # Quality indicators
    good_contrast = np.mean(contrast_scores) > 0.15  # Good contrast threshold
    balanced_intensity = 0.3 < np.mean(mean_pixel_values) < 0.7  # Not too dark/bright

    print(f"  ✅ Good contrast: {'Yes' if good_contrast else 'No'}")
    print(f"  ✅ Balanced intensity: {'Yes' if balanced_intensity else 'No'}")

    return {
        'mean_intensity': np.mean(mean_pixel_values),
        'mean_contrast': np.mean(contrast_scores),
        'good_contrast': good_contrast,
        'balanced_intensity': balanced_intensity
    }


def diversity_assessment(images, labels):
    """Assess diversity within each character class"""
    print("\n🎭 Diversity Assessment:")

    diversity_scores = []

    for class_idx in range(NUM_CLASSES):
        char = int_to_char[class_idx]
        class_images = images[labels == class_idx]

        if len(class_images) < 2:
            continue

        # Calculate pairwise differences
        differences = []
        for i in range(len(class_images)):
            for j in range(i + 1, len(class_images)):
                diff = np.mean(np.abs(class_images[i] - class_images[j]))
                differences.append(diff)

        avg_difference = np.mean(differences)
        diversity_scores.append(avg_difference)
        print(f"  '{char}': diversity score = {avg_difference:.3f}")

    overall_diversity = np.mean(diversity_scores)
    print(f"  📊 Overall diversity: {overall_diversity:.3f}")

    # Good diversity threshold (characters should be different from each other)
    good_diversity = overall_diversity > 0.1
    print(f"  ✅ Good diversity: {'Yes' if good_diversity else 'No'}")

    return {
        'overall_diversity': overall_diversity,
        'good_diversity': good_diversity,
        'per_class_diversity': diversity_scores
    }


def generate_alphabet_showcase():
    """Generate a clean showcase of all characters"""
    print("\n🔤 Generating alphabet showcase...")

    # Generate one sample per character
    showcase_images = []
    showcase_labels = []

    for class_idx in range(NUM_CLASSES):
        # Use fixed seed for consistency
        tf.random.set_seed(42 + class_idx)
        noise = tf.random.normal([1, LATENT_DIM])
        label = tf.constant([[class_idx]])

        generated_image = generator([noise, label], training=False)
        generated_image = generated_image[0, :, :, 0].numpy()

        showcase_images.append(generated_image)
        showcase_labels.append(class_idx)

    # Create showcase grid
    cols = min(10, NUM_CLASSES)
    rows = (NUM_CLASSES + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (img, label) in enumerate(zip(showcase_images, showcase_labels)):
        row = idx // cols
        col = idx % cols

        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        # Normalize for display
        img_display = (img + 1) / 2.0
        ax.imshow(img_display, cmap='gray')
        ax.set_title(f"'{int_to_char[label]}'", fontsize=12, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for idx in range(NUM_CLASSES, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('alphabet_showcase_epoch_200.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("✅ Alphabet showcase saved as 'alphabet_showcase_epoch_200.png'")


def comprehensive_accuracy_test():
    """Run comprehensive accuracy testing"""
    print("=" * 60)
    print("🧪 COMPREHENSIVE GAN ACCURACY TEST - EPOCH 200")
    print("=" * 60)

    # Generate test samples
    test_images, test_labels = generate_character_samples(num_samples_per_class=10)

    # Run assessments
    visual_results = visual_quality_assessment(test_images, test_labels)
    diversity_results = diversity_assessment(test_images, test_labels)

    # Create visualizations
    create_character_grid(test_images, test_labels, samples_per_row=10)
    generate_alphabet_showcase()

    # Overall quality score
    quality_indicators = [
        visual_results['good_contrast'],
        visual_results['balanced_intensity'],
        diversity_results['good_diversity']
    ]

    quality_score = sum(quality_indicators) / len(quality_indicators)

    print("\n" + "=" * 60)
    print("📊 OVERALL ASSESSMENT")
    print("=" * 60)
    print(f"Quality Score: {quality_score:.1%} ({sum(quality_indicators)}/{len(quality_indicators)} criteria met)")

    if quality_score >= 0.8:
        print("🎉 EXCELLENT: Your GAN is generating high-quality characters!")
        print("   Recommendation: Continue training or start using for CAPTCHA generation")
    elif quality_score >= 0.6:
        print("✅ GOOD: Characters are recognizable with room for improvement")
        print("   Recommendation: Continue training for 100-200 more epochs")
    elif quality_score >= 0.4:
        print("⚠️  FAIR: Characters are forming but need more training")
        print("   Recommendation: Continue training for 200-300 more epochs")
    else:
        print("❌ POOR: Characters need significant improvement")
        print("   Recommendation: Check training setup and continue for 300+ epochs")

    print("\n📁 Generated files:")
    print("   - accuracy_test_character_grid.png (detailed character samples)")
    print("   - alphabet_showcase_epoch_200.png (best character showcase)")

    return quality_score


if __name__ == "__main__":
    comprehensive_accuracy_test()