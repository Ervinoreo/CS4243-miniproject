from process_images import process_single_image
import os
import cv2
import numpy as np



def calculate_stroke_width_for_bbox(image, bbox):
    """
    Calculate the stroke width for a specific bounding box using simplified SWT approach.
    
    Args:
        image: Input BGR image
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
    
    Returns:
        float: Mean stroke width for the bounding box region, or 0 if no text detected
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    
    # Extract the ROI from the image
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    if roi.size == 0:
        return 0
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Invert the image so that text pixels are white
    inverted = cv2.bitwise_not(gray_roi)
    
    # Threshold to get binary image (assuming dark text on light background)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find stroke widths by looking at distances from text pixels to edges
    stroke_widths = []
    
    # Find contours to identify text regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter small contours (noise)
        if cv2.contourArea(contour) > 20:  # Minimum area threshold for bbox regions
            # Create mask for this contour
            mask = np.zeros(gray_roi.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Get distance values within this contour
            distances = dist_transform[mask > 0]
            if len(distances) > 0:
                # The stroke width is approximately 2 * distance transform value
                # (distance to nearest background pixel)
                stroke_width = np.mean(distances) * 2
                stroke_widths.append(stroke_width)
    
    if stroke_widths:
        return np.mean(stroke_widths)
    else:
        return 0

def filter_boxes_by_stroke_width(image, bounding_boxes, stroke_width_ratio_threshold=0.3):
    """
    Filter out bounding boxes with stroke widths significantly smaller than the median stroke width.
    
    Args:
        image: Input BGR image
        bounding_boxes: List of bounding box coordinates
        colors: List of colors corresponding to each bounding box
        stroke_width_ratio_threshold: Minimum ratio of box stroke width to median stroke width to keep (default: 0.3)
    
    Returns:
        Filtered bounding boxes and their corresponding colors
    """
    if not bounding_boxes:
        return [], []
    
    # Calculate stroke widths for all bounding boxes
    stroke_widths = []
    for bbox in bounding_boxes:
        stroke_width = calculate_stroke_width_for_bbox(image, bbox)
        stroke_widths.append(stroke_width)
    
    if not stroke_widths:
        return [], []
    
    # Calculate median stroke width (more robust than mean against outliers)
    valid_stroke_widths = [sw for sw in stroke_widths if sw > 0]
    if not valid_stroke_widths:
        # If no valid stroke widths detected, return all boxes
        return bounding_boxes
    
    median_stroke_width = np.median(valid_stroke_widths)
    min_stroke_threshold = median_stroke_width * stroke_width_ratio_threshold

    # print all stroke widths and median
    print("Stroke widths:", stroke_widths)
    print("Median stroke width:", median_stroke_width)
    
    # Filter boxes based on stroke width threshold
    filtered_boxes = []
    
    for bbox, stroke_width in zip(bounding_boxes, stroke_widths):
        # Keep boxes with stroke width >= threshold
        if stroke_width > 0 and stroke_width >= min_stroke_threshold:
            filtered_boxes.append(bbox)
    
    return filtered_boxes

path = os.path.join("data", "one", "0sw1nb-0.png")
image = cv2.imread(str(path))
_, valid_bboxes, _, wide_bboxes, _ = process_single_image(
    image_path=path, output_folder="output")

filtered_bboxes = filter_boxes_by_stroke_width(image, valid_bboxes, stroke_width_ratio_threshold=0.3)
print("Original boxes:", len(valid_bboxes))
print("Filtered boxes:", len(filtered_bboxes))
