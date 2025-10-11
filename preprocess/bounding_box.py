import cv2
import numpy as np
from collections import deque


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

def is_similar_color(color1, color2, threshold):
    """
    Check if two colors are similar within the threshold.
    
    Args:
        color1, color2: Colors in BGR format
        threshold: Color similarity threshold
    
    Returns:
        Boolean indicating if colors are similar
    """
    return np.linalg.norm(color1 - color2) <= threshold

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1, point2: Tuples of (x, y) coordinates
    
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_bounding_box_from_component(component, padding=0, image_shape=None):
    """Get bounding box from a connected component with padding."""
    if not component:
        return None
    
    y_coords = [pos[0] for pos in component]
    x_coords = [pos[1] for pos in component]
    
    y_min, y_max = min(y_coords), max(y_coords)
    x_min, x_max = min(x_coords), max(x_coords)
    
    if image_shape:
        height, width = image_shape
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width - 1, x_max + padding)
        y_max = min(height - 1, y_max + padding)
    else:
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = x_max + padding
        y_max = y_max + padding
    
    return (x_min, y_min, x_max, y_max)



def get_centroid_from_bbox(bbox):
    """Get centroid coordinates from bounding box."""
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return (center_x, center_y)


def is_bbox_nested(bbox1, bbox2):
    """Check if bbox1 is nested within bbox2."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    return (x2_min <= x1_min <= x1_max <= x2_max and 
            y2_min <= y1_min <= y1_max <= y2_max)


def get_bbox_area(bbox):
    """Calculate the area of a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)


def dfs_connected_component(mask, start_y, start_x, visited):
    """
    Perform DFS to find all connected pixels of the same component.
    
    Args:
        mask: Binary mask
        start_y, start_x: Starting position
        visited: Visited positions set
    
    Returns:
        List of (y, x) coordinates in the connected component
    """
    height, width = mask.shape
    component = []
    stack = deque([(start_y, start_x)])
    
    # 8-connectivity directions
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while stack:
        y, x = stack.pop()
        
        if (y, x) in visited or y < 0 or y >= height or x < 0 or x >= width:
            continue
            
        if not mask[y, x]:
            continue
            
        visited.add((y, x))
        component.append((y, x))
        
        # Add neighbors to stack
        for dy, dx in directions:
            new_y, new_x = y + dy, x + dx
            if (new_y, new_x) not in visited:
                stack.append((new_y, new_x))
    
    return component


def find_connected_components(mask, min_component_size=10):
    """
    Find all connected components in the mask using DFS.
    
    Args:
        mask: Binary mask
        min_component_size: Minimum size for a component to be considered
    
    Returns:
        List of components, each component is a list of (y, x) coordinates
    """
    height, width = mask.shape
    visited = set()
    components = []
    
    for y in range(height):
        for x in range(width):
            if mask[y, x] and (y, x) not in visited:
                component = dfs_connected_component(mask, y, x, visited)
                if len(component) >= min_component_size:
                    components.append(component)
    
    return components

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



def calculate_non_white_pixel_density(image, bbox, white_threshold=245):
    """
    Calculate the percentage of non-white pixels in a bounding box region.
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
        white_threshold: Threshold to consider a pixel as white
    
    Returns:
        Percentage of non-white pixels (0.0 to 1.0)
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Extract the region of interest
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    if roi.size == 0:
        return 0.0
    
    # Convert to grayscale for easier white detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Count non-white pixels (pixels below the white threshold)
    non_white_pixels = np.sum(gray_roi < white_threshold)
    total_pixels = gray_roi.size
    
    return non_white_pixels / total_pixels if total_pixels > 0 else 0.0


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


def draw_bounding_boxes(image, bounding_boxes, colors):
    """
    Draw bounding boxes on image for debugging.
    
    Args:
        image: Input image
        bounding_boxes: List of bounding box coordinates
        colors: List of colors corresponding to each box
    
    Returns:
        Image with bounding boxes drawn
    """
    debug_image = image.copy()
    
    for i, (bbox, color) in enumerate(zip(bounding_boxes, colors)):
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            # Draw rectangle with the detected color
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), color.tolist(), 2)
    
    return debug_image