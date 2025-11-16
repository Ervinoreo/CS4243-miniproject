import cv2
import numpy as np
from collections import deque
from color_analysis import is_similar_color



class UnionFind:
    """
    Union-Find Data Structure with path compression and union by rank.
    Used to efficiently merge similar color components that are nearby.
    """
    
    def __init__(self, n):
        """
        Initialize Union-Find structure.
        
        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, x):
        """
        Find the root of element x with path compression.
        
        Args:
            x: Element to find root for
        
        Returns:
            Root of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Union two sets containing x and y using union by rank.
        
        Args:
            x, y: Elements to union
        
        Returns:
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        return True
    
    def get_groups(self):
        """
        Get all groups as a dictionary mapping root to list of elements.
        
        Returns:
            Dictionary with root as key and list of elements as value
        """
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups
    
def get_centroid_from_bbox(bbox):
    """Get centroid coordinates from bounding box."""
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return (center_x, center_y)

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_bbox_area(bbox):
    """Calculate the area of a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

def is_bbox_nested(bbox1, bbox2):
    """Check if bbox1 is nested within bbox2."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    return (x2_min <= x1_min <= x1_max <= x2_max and 
            y2_min <= y1_min <= y1_max <= y2_max)

def merge_nearby_components(bounding_boxes, colors, color_threshold, distance_threshold):
    """Merge nearby connected components that have similar colors."""
    if not bounding_boxes:
        return [], []
    
    centroids = [get_centroid_from_bbox(bbox) for bbox in bounding_boxes]
    merged_boxes = []
    merged_colors = []
    used_indices = set()
    
    for i, (bbox1, color1, centroid1) in enumerate(zip(bounding_boxes, colors, centroids)):
        if i in used_indices:
            continue
            
        boxes_to_merge = [bbox1]
        used_indices.add(i)
        
        for j, (bbox2, color2, centroid2) in enumerate(zip(bounding_boxes, colors, centroids)):
            if j != i and j not in used_indices:
                color_similar = is_similar_color(color1, color2, color_threshold)
                distance = calculate_distance(centroid1, centroid2)
                distance_close = distance <= distance_threshold
                
                if color_similar and distance_close:
                    boxes_to_merge.append(bbox2)
                    used_indices.add(j)
        
        if boxes_to_merge:
            x_mins = [box[0] for box in boxes_to_merge]
            y_mins = [box[1] for box in boxes_to_merge]
            x_maxs = [box[2] for box in boxes_to_merge]
            y_maxs = [box[3] for box in boxes_to_merge]
            
            merged_box = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))
            merged_boxes.append(merged_box)
            merged_colors.append(color1)
    
    return merged_boxes, merged_colors


def merge_nested_components(bounding_boxes, colors, color_threshold, area_ratio_threshold=0.5):
    """Merge small bounding boxes that are nested within larger ones."""
    if not bounding_boxes:
        return [], []
    
    merged_boxes = []
    merged_colors = []
    used_indices = set()
    
    bbox_with_indices = [(i, bbox, color) for i, (bbox, color) in enumerate(zip(bounding_boxes, colors))]
    bbox_with_indices.sort(key=lambda x: get_bbox_area(x[1]), reverse=True)
    
    for i, (idx1, bbox1, color1) in enumerate(bbox_with_indices):
        if idx1 in used_indices:
            continue
            
        boxes_to_merge = [bbox1]
        colors_to_merge = [color1]
        used_indices.add(idx1)
        
        for j, (idx2, bbox2, color2) in enumerate(bbox_with_indices[i+1:], i+1):
            if idx2 in used_indices:
                continue
                
            is_nested = is_bbox_nested(bbox2, bbox1)
            is_color_similar = is_similar_color(color1, color2, color_threshold + 50)
            
            if is_nested and is_color_similar:
                area1 = get_bbox_area(bbox1)
                area2 = get_bbox_area(bbox2)
                area_ratio = area2 / area1 if area1 > 0 else 0
                
                if area_ratio <= area_ratio_threshold:
                    boxes_to_merge.append(bbox2)
                    colors_to_merge.append(color2)
                    used_indices.add(idx2)
        
        if boxes_to_merge:
            x_mins = [box[0] for box in boxes_to_merge]
            y_mins = [box[1] for box in boxes_to_merge]
            x_maxs = [box[2] for box in boxes_to_merge]
            y_maxs = [box[3] for box in boxes_to_merge]
            
            merged_box = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))
            merged_boxes.append(merged_box)
            merged_colors.append(color1)
    
    return merged_boxes, merged_colors



def merge_components_with_ufds(bounding_boxes, colors, color_threshold, distance_threshold):
    """
    Merge nearby connected components with similar colors using Union-Find Data Structure.
    This is more efficient than the previous approach for large numbers of components.
    
    Args:
        bounding_boxes: List of bounding box coordinates
        colors: List of colors corresponding to each bounding box
        color_threshold: Threshold for color similarity
        distance_threshold: Maximum distance between centroids to consider merging
    
    Returns:
        Merged bounding boxes and their corresponding colors
    """
    if not bounding_boxes:
        return [], []
    
    n = len(bounding_boxes)
    if n == 1:
        return bounding_boxes, colors
    
    # Initialize Union-Find structure
    uf = UnionFind(n)
    
    # Calculate centroids for all bounding boxes
    centroids = [get_centroid_from_bbox(bbox) for bbox in bounding_boxes]
    
    # Find pairs that should be merged and union them
    for i in range(n):
        for j in range(i + 1, n):
            # Check if colors are similar
            color_similar = is_similar_color(colors[i], colors[j], color_threshold)
            
            # Check if centroids are close
            distance = calculate_distance(centroids[i], centroids[j])
            distance_close = distance <= distance_threshold
            
            # Union if both conditions are met
            if color_similar and distance_close:
                uf.union(i, j)
    
    # Get groups of components that should be merged
    groups = uf.get_groups()
    
    merged_boxes = []
    merged_colors = []
    
    # Merge components in each group
    for root, group_indices in groups.items():
        if len(group_indices) == 1:
            # Single component, no merging needed
            idx = group_indices[0]
            merged_boxes.append(bounding_boxes[idx])
            merged_colors.append(colors[idx])
        else:
            # Multiple components to merge
            boxes_to_merge = [bounding_boxes[idx] for idx in group_indices]
            colors_to_merge = [colors[idx] for idx in group_indices]
            
            # Merge bounding boxes
            x_mins = [box[0] for box in boxes_to_merge]
            y_mins = [box[1] for box in boxes_to_merge]
            x_maxs = [box[2] for box in boxes_to_merge]
            y_maxs = [box[3] for box in boxes_to_merge]
            
            merged_box = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))
            merged_boxes.append(merged_box)
            
            # Use the color from the largest component in the group
            areas = [get_bbox_area(bounding_boxes[idx]) for idx in group_indices]
            largest_idx = group_indices[np.argmax(areas)]
            merged_colors.append(colors[largest_idx])
    
    return merged_boxes, merged_colors



def filter_boxes_by_size(bounding_boxes, colors, size_ratio_threshold=0.5, large_box_ratio=2.0):
    """
    Filter out bounding boxes that are significantly smaller or larger than the average.
    
    Args:
        bounding_boxes: List of bounding box coordinates
        colors: List of colors corresponding to each bounding box
        size_ratio_threshold: Minimum ratio of box area to median area to keep (default: 0.5)
        large_box_ratio: Maximum ratio of box area to median area to keep (default: 2.0)
    
    Returns:
        Filtered bounding boxes and their corresponding colors
    """
    if not bounding_boxes:
        return [], []
    
    # Calculate areas for all bounding boxes
    areas = [get_bbox_area(bbox) for bbox in bounding_boxes]
    
    if not areas:
        return [], []
    
    # Use median area as reference (more robust than mean against outliers)
    median_area = np.median(areas)
    min_area_threshold = median_area * size_ratio_threshold
    max_area_threshold = median_area * large_box_ratio
    
    # Filter boxes based on area thresholds
    filtered_boxes = []
    filtered_colors = []
    
    for bbox, color, area in zip(bounding_boxes, colors, areas):
        if min_area_threshold <= area <= max_area_threshold:
            filtered_boxes.append(bbox)
            filtered_colors.append(color)
    
    return filtered_boxes, filtered_colors


def filter_boxes_by_pixel_density(image, bounding_boxes, colors, density_ratio_threshold=0.25, white_threshold=245):
    """Filter out bounding boxes with significantly lower non-white pixel count."""
    if not bounding_boxes:
        return [], []
    
    pixel_counts = []
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        roi = image[y_min:y_max+1, x_min:x_max+1]
        
        if roi.size == 0:
            pixel_counts.append(0)
            continue
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        non_white_pixels = np.sum(gray_roi < white_threshold)
        pixel_counts.append(non_white_pixels)
    
    if not pixel_counts:
        return [], []
    
    median_pixel_count = np.median(pixel_counts)
    min_pixel_threshold = median_pixel_count * density_ratio_threshold
    
    filtered_boxes = []
    filtered_colors = []
    
    for bbox, color, pixel_count in zip(bounding_boxes, colors, pixel_counts):
        if pixel_count >= min_pixel_threshold:
            filtered_boxes.append(bbox)
            filtered_colors.append(color)
    
    return filtered_boxes, filtered_colors



def filter_large_colored_boxes(bounding_boxes, colors, box_types, all_bboxes, size_multiplier=2.0):
    """Filter out large colored bounding boxes based on size comparison with all boxes."""
    if not bounding_boxes or not all_bboxes:
        return bounding_boxes, colors, box_types
    
    # Calculate areas of all bounding boxes
    all_areas = []
    for bbox in all_bboxes:
        if len(bbox) >= 4:  # Ensure bbox has at least 4 elements (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = bbox[:4]
            area = (x_max - x_min) * (y_max - y_min)
            all_areas.append(area)
    
    if not all_areas:
        return bounding_boxes, colors, box_types
    
    # Calculate mean and median areas
    mean_area = np.mean(all_areas)
    median_area = np.median(all_areas)
    threshold_area = max(mean_area, median_area) * size_multiplier
    
    # Filter colored boxes that are too large
    filtered_boxes = []
    filtered_colors = []
    filtered_types = []
    
    for bbox, color, box_type in zip(bounding_boxes, colors, box_types):
        x_min, y_min, x_max, y_max = bbox[:4]
        area = (x_max - x_min) * (y_max - y_min)
        
        # Only filter colored boxes (not DFS boxes)
        if box_type == 'color' and area > threshold_area:
            continue  # Skip this large colored box
        
        filtered_boxes.append(bbox)
        filtered_colors.append(color)
        filtered_types.append(box_type)
    
    return filtered_boxes, filtered_colors, filtered_types

def calculate_stroke_width_transform(gray_image: np.ndarray):
    """
    Calculate Mean Stroke Width using a simplified SWT approach.
    
    Args:
        gray_image (np.ndarray): Grayscale image
        
    Returns:
        Dict[str, float]: Dictionary containing stroke width statistics
    """
    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Apply morphological operations to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Calculate distance transform
    # Invert the image so that text pixels are white
    inverted = cv2.bitwise_not(gray_image)
    
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
        if cv2.contourArea(contour) > 50:  # Minimum area threshold
            # Create mask for this contour
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Get distance values within this contour
            distances = dist_transform[mask > 0]
            if len(distances) > 0:
                # The stroke width is approximately 2 * distance transform value
                # (distance to nearest background pixel)
                stroke_width = np.mean(distances) * 2
                stroke_widths.append(stroke_width)
    
    if stroke_widths:
        mean_stroke_width = np.mean(stroke_widths)
        std_stroke_width = np.std(stroke_widths)
        median_stroke_width = np.median(stroke_widths)
        
        # Classify stroke thickness
        if mean_stroke_width < 3:
            thickness_level = "Very Thin"
        elif mean_stroke_width < 5:
            thickness_level = "Thin"
        elif mean_stroke_width < 8:
            thickness_level = "Medium"
        elif mean_stroke_width < 12:
            thickness_level = "Thick"
        else:
            thickness_level = "Very Thick"
    else:
        mean_stroke_width = 0
        std_stroke_width = 0
        median_stroke_width = 0
        thickness_level = "No Text Detected"
    
    return {
        'mean_stroke_width': mean_stroke_width,
        'std_stroke_width': std_stroke_width,
        'median_stroke_width': median_stroke_width,
        'thickness_level': thickness_level,
        'num_text_regions': len(stroke_widths)
    }

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

    stroke_width_info = calculate_stroke_width_transform(gray_roi)

    return stroke_width_info['mean_stroke_width']


def filter_boxes_by_stroke_width(image, valid_bboxes, char_bboxes, char_colors, 
                                 stroke_width_ratio_threshold=0.1, wide_box_color_threshold=30):
    """
    Filter bounding boxes based on stroke width compared to median stroke width.
    
    Args:
        image: Input BGR image
        valid_bboxes: List of valid bounding boxes (DFS format with 6 elements)
        char_bboxes: List of character bounding boxes (4 elements format)
        char_colors: List of colors corresponding to character bounding boxes
        stroke_width_ratio_threshold: Maximum ratio of box stroke width to median stroke width to keep boxes
        wide_box_color_threshold: Color threshold for masking when processing character boxes
    
    Returns:
        Tuple containing (filtered_valid_bboxes, filtered_char_bboxes, filtered_char_colors)
    """
    from color_analysis import apply_color_mask_to_segment
    
    stroke_width_filtered_valid_bboxes = []
    stroke_width_filtered_char_bboxes = []
    stroke_width_filtered_char_colors = []
    
    # Calculate stroke widths for all bounding boxes
    all_stroke_widths = []
    
    # Calculate stroke widths for valid bboxes (no color masking needed)
    valid_stroke_widths = []
    for valid_bbox in valid_bboxes:
        x1, y1, x2, y2, _, _ = valid_bbox
        bbox_coords = (x1, y1, x2, y2)
        stroke_width = calculate_stroke_width_for_bbox(image, bbox_coords)
        valid_stroke_widths.append(stroke_width)
        all_stroke_widths.append(stroke_width)
    
    # Calculate stroke widths for character bboxes (apply color masking first)
    char_stroke_widths = []
    for char_bbox, char_color in zip(char_bboxes, char_colors):
        x1, y1, x2, y2 = char_bbox
        
        # Extract the segment
        segment = image[y1:y2+1, x1:x2+1]
        
        if segment.size > 0:
            # Apply color masking if a specific color was detected
            if char_color is not None:
                masked_segment = apply_color_mask_to_segment(segment, char_color, wide_box_color_threshold)
            else:
                masked_segment = segment
            
            # Calculate stroke width on the masked segment
            stroke_width = calculate_stroke_width_for_bbox(masked_segment, (0, 0, masked_segment.shape[1]-1, masked_segment.shape[0]-1))
        else:
            stroke_width = 0
        
        char_stroke_widths.append(stroke_width)
        all_stroke_widths.append(stroke_width)
    
    # Calculate median stroke width for filtering
    if all_stroke_widths:
        median_stroke_width = np.median([sw for sw in all_stroke_widths if sw > 0])
        stroke_width_threshold = median_stroke_width * stroke_width_ratio_threshold
        
        # Filter valid bboxes based on stroke width
        for valid_bbox, stroke_width in zip(valid_bboxes, valid_stroke_widths):
            stroke_width_filtered_valid_bboxes.append(valid_bbox)
        
        # Filter character bboxes based on stroke width
        for char_bbox, char_color, stroke_width in zip(char_bboxes, char_colors, char_stroke_widths):
            if stroke_width > 0 and stroke_width > stroke_width_threshold:
                stroke_width_filtered_char_bboxes.append(char_bbox)
                stroke_width_filtered_char_colors.append(char_color)
    else:
        # No valid stroke widths calculated, keep all boxes
        stroke_width_filtered_valid_bboxes = valid_bboxes
        stroke_width_filtered_char_bboxes = char_bboxes
        stroke_width_filtered_char_colors = char_colors
    
    return stroke_width_filtered_valid_bboxes, stroke_width_filtered_char_bboxes, stroke_width_filtered_char_colors
