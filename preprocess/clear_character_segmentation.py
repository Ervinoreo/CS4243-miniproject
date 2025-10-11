"""
Character Segmentation Module

This module handles the final character segmentation process.
"""

import cv2
import numpy as np
from color_analysis import is_similar_color, find_majority_color


def segment_character(image, bbox, detected_color, color_threshold, detected_color_threshold=30, padding=10, black_threshold=30):
    """
    Segment a character from the image based on bounding box using the detected color.
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
        detected_color: The detected color for this character (from color analysis)
        color_threshold: Threshold for color similarity
        detected_color_threshold: Threshold for detecting the target color (default: 30)
        padding: Padding around the character in pixels
        black_threshold: Threshold for filtering out black pixels
    
    Returns:
        Segmented character image with padding
    """
    if bbox is None:
        return None
    
    # Extract the region with padding
    char_region = extract_region_with_padding(image, bbox, padding)
    if char_region is None:
        return None
    
    # Use the detected color directly instead of finding majority color
    target_color = detected_color
    
    # Create output image - white background
    output_image = np.full_like(char_region, 255, dtype=np.uint8)
    
    # Keep only pixels similar to the detected color
    region_height, region_width = char_region.shape[:2]
    for y in range(region_height):
        for x in range(region_width):
            pixel_color = char_region[y, x]
            # Skip very dark pixels (black lines)
            if np.sum(pixel_color) <= black_threshold:
                continue
            if is_similar_color(pixel_color, target_color, detected_color_threshold):
                output_image[y, x] = pixel_color
    
    return output_image


def create_character_mask(char_region, detected_color, detected_color_threshold=30, black_threshold=30):
    """
    Create a mask for character pixels based on the detected color.
    
    Args:
        char_region: Character region image
        detected_color: The detected color for this character
        detected_color_threshold: Threshold for detecting the target color
        black_threshold: Threshold for filtering out black pixels
    
    Returns:
        Binary mask for character pixels
    """
    height, width = char_region.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            pixel_color = char_region[y, x]
            # Skip very dark pixels (black lines)
            if np.sum(pixel_color) <= black_threshold:
                continue
            if is_similar_color(pixel_color, detected_color, detected_color_threshold):
                mask[y, x] = True
    
    return mask


def apply_padding(x_min, y_min, x_max, y_max, padding, image_shape):
    """
    Apply padding to bounding box coordinates with boundary checking.
    
    Args:
        x_min, y_min, x_max, y_max: Original bounding box coordinates
        padding: Padding to apply in pixels
        image_shape: (height, width) of the image for boundary checking
    
    Returns:
        Padded coordinates (x_min, y_min, x_max, y_max)
    """
    height, width = image_shape
    padded_x_min = max(0, x_min - padding)
    padded_y_min = max(0, y_min - padding)
    padded_x_max = min(width - 1, x_max + padding)
    padded_y_max = min(height - 1, y_max + padding)
    
    return padded_x_min, padded_y_min, padded_x_max, padded_y_max


def extract_region_with_padding(image, bbox, padding=10):
    """
    Extract a region from the image with padding applied to the bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
        padding: Padding around the region in pixels
    
    Returns:
        Extracted region with padding applied
    """
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    height, width = image.shape[:2]
    
    # Apply padding to bounding box coordinates
    padded_x_min, padded_y_min, padded_x_max, padded_y_max = apply_padding(
        x_min, y_min, x_max, y_max, padding, (height, width)
    )
    
    # Extract the region with padding
    region = image[padded_y_min:padded_y_max+1, padded_x_min:padded_x_max+1].copy()
    
    return region