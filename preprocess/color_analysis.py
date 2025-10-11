"""
Color Analysis Module

This module handles color extraction and analysis operations for character segmentation.
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def extract_main_colors_dbscan(image, eps=15, min_samples=50, threshold=0.005, white_threshold=250, black_threshold=30):
    """Extract main colors from image using DBSCAN clustering."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    
    pixel_sums = np.sum(pixels, axis=1)
    valid_pixels = pixels[(pixel_sums < white_threshold * 3) & (pixel_sums > black_threshold)]
    
    if len(valid_pixels) == 0:
        return []
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(valid_pixels)
    
    main_colors = []
    total_pixels = len(valid_pixels)
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            continue
            
        cluster_pixels = valid_pixels[labels == label]
        cluster_size = len(cluster_pixels)
        
        if cluster_size / total_pixels >= threshold:
            representative_color = np.median(cluster_pixels, axis=0)
            color_bgr = representative_color[::-1]
            main_colors.append(color_bgr.astype(np.uint8))
    
    return main_colors


def is_similar_color(color1, color2, threshold):
    """Check if two colors are similar within the threshold."""
    return np.linalg.norm(color1 - color2) <= threshold


def create_color_mask_for_color(image, target_color, color_threshold):
    """Create a binary mask for pixels similar to target color."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            pixel_color = image[y, x]
            if is_similar_color(pixel_color, target_color, color_threshold):
                mask[y, x] = True
    
    return mask


def find_majority_color(pixels, black_threshold=30):
    """
    Find the most common color in a set of pixels, excluding very dark pixels.
    
    Args:
        pixels: Array of pixel colors
        black_threshold: Threshold for filtering out black pixels
    
    Returns:
        Most common color (BGR format)
    """
    # Remove white pixels and very dark pixels (remaining black lines)
    white_threshold_local = 245  # Slightly more lenient than before
    
    # Filter out white and very black pixels
    pixel_sums = np.sum(pixels, axis=1)
    valid_pixels = pixels[(pixel_sums < white_threshold_local * 3) & (pixel_sums > black_threshold)]
    
    if len(valid_pixels) == 0:
        return None
    
    # Find the most common valid color
    unique_colors, counts = np.unique(valid_pixels, axis=0, return_counts=True)
    majority_color = unique_colors[np.argmax(counts)]
    
    return majority_color


def create_color_mask(image, target_color, color_threshold):
    """
    Create a binary mask for pixels similar to target color.
    
    Args:
        image: Input image (BGR format)
        target_color: Target color (BGR format)
        color_threshold: Threshold for color similarity
    
    Returns:
        Binary mask where True indicates pixels similar to target color
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            pixel_color = image[y, x]
            if is_similar_color(pixel_color, target_color, color_threshold):
                mask[y, x] = True
    
    return mask