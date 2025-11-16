import cv2
import numpy as np

def dfs_connected_component_black_white(mask, visited, start_row, start_col, component_pixels, black_threshold):
    """
    Depth-First Search to find all pixels in a connected component.
    
    Args:
        mask: Grayscale mask image
        visited: Boolean array to track visited pixels
        start_row: Starting row coordinate
        start_col: Starting column coordinate
        component_pixels: List to store pixels belonging to this component
        black_threshold: Threshold below which pixels are considered black
    """
    rows, cols = mask.shape
    
    # Stack for DFS (using list as stack)
    stack = [(start_row, start_col)]
    
    # 8-connected neighborhood directions
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while stack:
        row, col = stack.pop()
        
        # Skip if already visited or out of bounds
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            visited[row, col] or mask[row, col] <= black_threshold):
            continue
        
        # Mark as visited and add to component
        visited[row, col] = True
        component_pixels.append((row, col))
        
        # Add all 8-connected neighbors to stack
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                not visited[new_row, new_col] and mask[new_row, new_col] > black_threshold):
                stack.append((new_row, new_col))

def draw_bounding_boxes_using_dfs(mask, min_area=40, black_threshold=5, width_threshold=1.1):
    """
    Find connected components using DFS and draw bounding boxes around them.
    
    Args:
        mask: Binary mask image
        min_area: Minimum area threshold for connected components (default: 40)
        black_threshold: Threshold below which pixels are considered black (default: 5)
        width_threshold: Threshold for filtering wide bounding boxes (width > height * threshold) (default: 1.1)
    
    Returns:
        mask_with_boxes: Mask image with bounding boxes drawn
        num_components: Number of valid (non-wide) connected components found
        valid_bboxes: List of valid bounding boxes [(x1, y1, x2, y2, width, height), ...]
        wide_bboxes: List of wide bounding boxes [(x1, y1, x2, y2, width, height), ...]
    """
    # Convert to grayscale if not already
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    
    # Create output image (convert to 3-channel if needed)
    if len(mask.shape) == 3:
        mask_with_boxes = mask.copy()
    else:
        mask_with_boxes = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    rows, cols = mask_gray.shape
    visited = np.zeros((rows, cols), dtype=bool)
    valid_components = 0
    valid_bboxes = []
    wide_bboxes = []
    
    # Scan through the image to find connected components using DFS
    for row in range(rows):
        for col in range(cols):
            # If we find a non-black pixel that hasn't been visited
            if mask_gray[row, col] > black_threshold and not visited[row, col]:
                component_pixels = []
                
                # Use DFS to find all connected pixels
                dfs_connected_component_black_white(mask_gray, visited, row, col, component_pixels, black_threshold)
                
                # Check if component is large enough
                if len(component_pixels) >= min_area:
                    # Calculate bounding box coordinates
                    rows_coords = [pixel[0] for pixel in component_pixels]
                    cols_coords = [pixel[1] for pixel in component_pixels]
                    
                    min_row, max_row = min(rows_coords), max(rows_coords)
                    min_col, max_col = min(cols_coords), max(cols_coords)
                    
                    # Calculate width and height
                    width = max_col - min_col + 1
                    height = max_row - min_row + 1
                    
                    # Store bounding box information (x1, y1, x2, y2, width, height)
                    bbox_info = (min_col, min_row, max_col, max_row, width, height)
                    
                    # Check if bounding box is too wide
                    is_too_wide = width > height * width_threshold
                    
                    if is_too_wide:
                        # Add to wide bounding boxes array
                        wide_bboxes.append(bbox_info)
                        # Draw wide bounding box in red with "W" mark
                        cv2.rectangle(mask_with_boxes, (min_col, min_row), (max_col, max_row), (0, 0, 255), 2)
                        cv2.putText(mask_with_boxes, "W", 
                                   (min_col, min_row - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # Add to valid bounding boxes array
                        valid_bboxes.append(bbox_info)
                        # Draw valid bounding box in green
                        cv2.rectangle(mask_with_boxes, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                        # Draw component number
                        cv2.putText(mask_with_boxes, str(valid_components + 1), 
                                   (min_col, min_row - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        valid_components += 1
    
    return mask_with_boxes, valid_components, valid_bboxes, wide_bboxes


def find_color_connected_components_dfs(mask, min_component_size=10):
    """Find all connected components in the mask using DFS."""
    height, width = mask.shape
    visited = set()
    components = []
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y in range(height):
        for x in range(width):
            if mask[y, x] and (y, x) not in visited:
                component = []
                stack = [(y, x)]
                
                while stack:
                    cy, cx = stack.pop()
                    if (cy, cx) in visited or cy < 0 or cy >= height or cx < 0 or cx >= width:
                        continue
                    if not mask[cy, cx]:
                        continue
                    
                    visited.add((cy, cx))
                    component.append((cy, cx))
                    
                    for dy, dx in directions:
                        new_y, new_x = cy + dy, cx + dx
                        if (new_y, new_x) not in visited:
                            stack.append((new_y, new_x))
                
                if len(component) >= min_component_size:
                    components.append(component)
    
    return components


def draw_bounding_box_from_component(component, padding=0, image_shape=None):
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