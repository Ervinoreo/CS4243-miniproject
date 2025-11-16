import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def is_similar_color(color1, color2, threshold):
    """Check if two colors are similar within the threshold."""
    return np.linalg.norm(color1 - color2) <= threshold

def create_color_mask_black_white(image, white_threshold=200, black_threshold=50):
    """
    Create a mask for colored parts of the image by excluding white and black regions.
    
    Args:
        image: Input BGR image
        white_threshold: Threshold above which pixels are considered white (default: 200)
        black_threshold: Threshold below which pixels are considered black (default: 50)
    
    Returns:
        mask: Binary mask where colored regions are white (255) and non-colored are black (0)
    """
    # Convert BGR to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create masks for white and black regions
    white_mask = gray > white_threshold
    black_mask = gray < black_threshold
    
    # Create colored region mask (not white and not black)
    colored_mask = ~(white_mask | black_mask)
    
    # Convert boolean mask to uint8 (0 or 255)
    colored_mask = colored_mask.astype(np.uint8) * 255
    
    return colored_mask


def create_color_mask_white(image, white_threshold=250):
    """
    Create a mask for colored parts of the image by excluding white regions.

    Args:
        image: Input BGR image
        white_threshold: Threshold above which pixels are considered white (default: 250)
    
    Returns:
        mask: Binary mask where colored regions are white (255) and non-colored are black (0)
    """
    # Convert BGR to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create masks for white and black regions
    white_mask = gray > white_threshold
    
    # Create colored region mask (not white)
    colored_mask = ~(white_mask)
    
    # Convert boolean mask to uint8 (0 or 255)
    colored_mask = colored_mask.astype(np.uint8) * 255
    
    return colored_mask

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

def smooth_mask(mask, kernel_size=3, stride=3):
    """
    Apply smoothening technique to the mask using average (box) filter.
    
    Args:
        mask: Binary mask image
        kernel_size: Size of the averaging kernel (default: 3)
    
    Returns:
        smoothed_mask: Smoothened binary mask
    """
    # Apply average blur (box filter) to smooth the mask with stride
    blurred = cv2.blur(mask, (kernel_size, kernel_size))
    if stride > 1:
        blurred = blurred[::stride, ::stride]

        # Upsample the blurred mask back to original size
        blurred = cv2.resize(blurred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    return blurred


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

def check_multiple_colors_in_bbox(image, bbox, color_difference_threshold=50.0, white_threshold=245, black_threshold=10):
    """
    Check if a bounding box contains multiple distinct colors using DBSCAN.
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box tuple (x1, y1, x2, y2, width, height)
        color_difference_threshold: Threshold for determining if colors are different enough
        white_threshold: Threshold for white pixels
        black_threshold: Threshold for black pixels
    
    Returns:
        Boolean indicating if the box contains multiple distinct colors
    """
    x1, y1, x2, y2, _, _ = bbox
    
    # Extract the ROI from the image
    roi = image[y1:y2+1, x1:x2+1]
    
    if roi.size == 0:
        return False
    
    # Use DBSCAN to find main colors in the ROI
    main_colors = extract_main_colors_dbscan(
        roi, 
        eps=15, 
        min_samples=20,  # Lower threshold for smaller regions
        threshold=0.01,  # Lower threshold for smaller regions
        white_threshold=white_threshold, 
        black_threshold=black_threshold
    )
    
    # If we have less than 2 colors, it's not multi-colored
    if len(main_colors) < 2:
        return False
    
    # Check if any two colors are different enough
    for i in range(len(main_colors)):
        for j in range(i + 1, len(main_colors)):
            color_distance = np.linalg.norm(main_colors[i].astype(float) - main_colors[j].astype(float))
            if color_distance >= color_difference_threshold:
                return True
    
    return False


def apply_color_mask_to_segment(segment, target_color, color_mask_threshold=70):
    """
    Apply color masking to a segment, making non-matching pixels white.
    
    Args:
        segment: Image segment (BGR format)
        target_color: Target color to preserve (BGR format)
        color_mask_threshold: Threshold for color similarity
    
    Returns:
        Masked segment with non-matching pixels set to white
    """
    if segment.size == 0:
        return segment
    
    # Create a copy of the segment
    masked_segment = segment.copy()
    
    # Calculate color differences for each pixel
    height, width = segment.shape[:2]
    for y in range(height):
        for x in range(width):
            pixel_color = segment[y, x]
            # Calculate Euclidean distance in color space
            color_distance = np.linalg.norm(pixel_color.astype(float) - target_color.astype(float))
            
            # If pixel is not similar to target color, make it white
            if color_distance > color_mask_threshold:
                masked_segment[y, x] = [255, 255, 255]  # White in BGR
    
    return masked_segment


def apply_black_threshold_to_segment(segment, black_threshold):
    """
    Apply black threshold filtering to a valid segment, making pixels below threshold white.
    
    Args:
        segment: Image segment (BGR format)
        black_threshold: Threshold below which pixels are made white
    
    Returns:
        Processed segment with dark pixels below threshold set to white
    """
    if segment.size == 0:
        return segment
    
    # Create a copy of the segment
    processed_segment = segment.copy()
    
    # Convert to grayscale to check pixel intensity
    gray_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    
    # Create mask for pixels below black threshold
    below_threshold_mask = gray_segment < black_threshold
    
    # Set pixels below threshold to white
    processed_segment[below_threshold_mask] = [255, 255, 255]  # White in BGR
    
    return processed_segment
