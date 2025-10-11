import cv2
import numpy as np
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from sklearn.cluster import DBSCAN
from multiprocessing import Pool, cpu_count
from functools import partial
from detect_connected_components import create_color_mask, smooth_mask, draw_bounding_boxes_using_dfs
from color_analysis import extract_main_colors_dbscan, is_similar_color, create_color_mask_for_color
from bounding_box import (
    get_bounding_box_from_component,
    merge_nearby_components,
    merge_nested_components,
    merge_components_with_ufds,
    filter_boxes_by_pixel_density,
    filter_boxes_by_size,
    filter_large_colored_boxes
)
from utils import save_unclear_processing_parameters_to_json

def find_connected_components_dfs(mask, min_component_size=10):
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

def process_wide_bounding_box(image, wide_bbox, color_threshold=40, padding=3, distance_threshold=30, area_ratio_threshold=0.2, density_ratio_threshold=0.25, white_threshold=245, black_threshold=10):
    """Process a wide bounding box to extract character components using DBSCAN."""
    x_min, y_min, x_max, y_max, width, height = wide_bbox
    
    # Extract the ROI from the original image
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    if roi.size == 0:
        return [], []
    
    # Step 1: Use DBSCAN to find colored parts
    main_colors = extract_main_colors_dbscan(
        roi, 
        eps=15, 
        min_samples=50, 
        threshold=0.005,
        white_threshold=white_threshold, 
        black_threshold=black_threshold
    )
    
    if not main_colors:
        return [], []
    
    # Step 2: Find connected components for each color
    all_bounding_boxes = []
    all_colors = []
    
    for color in main_colors:
        color_mask = create_color_mask_for_color(roi, color, color_threshold)
        components = find_connected_components_dfs(color_mask, min_component_size=1)
        
        for component in components:
            bbox = get_bounding_box_from_component(
                component, 
                padding=padding, 
                image_shape=roi.shape[:2]
            )
            if bbox is not None:
                # Convert bbox coordinates back to original image coordinates
                global_bbox = (bbox[0] + x_min, bbox[1] + y_min, 
                              bbox[2] + x_min, bbox[3] + y_min)
                all_bounding_boxes.append(global_bbox)
                all_colors.append(color)
    
    if not all_bounding_boxes:
        return [], []
    
    # Step 3: Merge nearby components with similar colors using UFDS
    ufds_merged_boxes, ufds_merged_colors = merge_components_with_ufds(
        all_bounding_boxes, all_colors, color_threshold, distance_threshold
    )
    
    # Step 4: Merge nearby components with similar colors using distance-based method
    nearby_merged_boxes, nearby_merged_colors = merge_nearby_components(
        ufds_merged_boxes, ufds_merged_colors, color_threshold, distance_threshold
    )
    
    # Step 5: Merge nested components with similar colors
    nested_merged_boxes, nested_merged_colors = merge_nested_components(
        nearby_merged_boxes, nearby_merged_colors, color_threshold, area_ratio_threshold
    )
    
    if not nested_merged_boxes:
        return [], []
    
    # Step 6: Filter out low pixel density boxes
    final_boxes, final_colors = filter_boxes_by_pixel_density(
        image, nested_merged_boxes, nested_merged_colors, density_ratio_threshold, white_threshold
    )
    
    return final_boxes, final_colors


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


def process_single_image(image_path, output_folder, white_threshold=200, black_threshold=50, kernel_size=3, stride=3, min_area=100, width_threshold=1.1, segment_padding=3, color_mask_threshold=70, wide_box_color_threshold=30, size_multiplier=2.0, size_ratio_threshold=0.5, large_box_ratio=2.0, color_difference_threshold=50.0):
    """
    Process a single image to create combined visualization with original image, mask, and smoothened mask with bounding boxes.
    Also segments the image based on bounding boxes.
    
    Args:
        image_path: Path to input image
        output_folder: Path to output folder
        white_threshold: White threshold value
        black_threshold: Black threshold value
        kernel_size: Kernel size for smoothening
        stride: Stride for smoothening
        min_area: Minimum area for connected components
        width_threshold: Threshold for filtering wide bounding boxes (width > height * threshold)
        segment_padding: Padding for image segmentation
        color_mask_threshold: Threshold for color masking in character segments
        wide_box_color_threshold: Threshold for color masking when segmenting wide boxes
        size_ratio_threshold: Minimum ratio of box area to median area to keep boxes
        large_box_ratio: Maximum ratio of box area to median area to keep boxes
        color_difference_threshold: Color difference threshold for reclassifying valid boxes as wide boxes
    
    Returns:
        Tuple containing (success_status, valid_bboxes, wide_bboxes)
    """
    try:
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return False, [], [], [], []
        
        print(f"Processing: {image_path.name}")
        
        # Create color mask
        mask = create_color_mask(image, white_threshold, black_threshold)
        
        # Apply averaging to the mask
        smoothed_mask = smooth_mask(mask, kernel_size, stride)
        
        # Draw bounding boxes on smoothed mask
        smoothed_mask_with_boxes, num_components, valid_bboxes, wide_bboxes = draw_bounding_boxes_using_dfs(smoothed_mask, min_area, black_threshold, width_threshold)
        
        # Check valid boxes for multiple colors and reclassify as wide boxes if needed
        remaining_valid_bboxes = []
        additional_wide_bboxes = []
        
        for valid_bbox in valid_bboxes:
            if check_multiple_colors_in_bbox(image, valid_bbox, color_difference_threshold, white_threshold, black_threshold):
                # Reclassify as wide box
                additional_wide_bboxes.append(valid_bbox)
            else:
                # Keep as valid box
                remaining_valid_bboxes.append(valid_bbox)
        
        # Update the lists
        valid_bboxes = remaining_valid_bboxes
        wide_bboxes.extend(additional_wide_bboxes)
        
        # Process wide bounding boxes to extract character components
        all_wide_char_bboxes = []
        all_wide_char_colors = []
        
        for wide_bbox in wide_bboxes:
            char_bboxes, char_colors = process_wide_bounding_box(
                image, wide_bbox,
                color_threshold=30,
                padding=3,
                distance_threshold=30,
                area_ratio_threshold=0.2,
                density_ratio_threshold=0.25,
                white_threshold=white_threshold,
                black_threshold=black_threshold
            )
            
            # If no character components are detected, use the original wide box
            if not char_bboxes:
                # Convert wide_bbox format (x_min, y_min, x_max, y_max, width, height) to (x1, y1, x2, y2)
                x_min, y_min, x_max, y_max, width, height = wide_bbox
                char_bboxes = [(x_min, y_min, x_max, y_max)]
                char_colors = [None]  # No specific color detected
            
            all_wide_char_bboxes.extend(char_bboxes)
            all_wide_char_colors.extend(char_colors)
        
        # Combine all bounding boxes and their types for filtering
        all_combined_bboxes = []
        all_combined_colors = []
        all_combined_types = []
        
        # Add valid bboxes (from DFS)
        for valid_bbox in valid_bboxes:
            all_combined_bboxes.append(valid_bbox[:4])  # Take only x1, y1, x2, y2
            all_combined_colors.append(None)  # No specific color for DFS boxes
            all_combined_types.append('dfs')
        
        # Add character bboxes (from color detection)
        for char_bbox, char_color in zip(all_wide_char_bboxes, all_wide_char_colors):
            all_combined_bboxes.append(char_bbox)
            all_combined_colors.append(char_color)
            all_combined_types.append('color')
        
        # Filter out large colored boxes
        filtered_bboxes, filtered_colors, filtered_types = filter_large_colored_boxes(
            all_combined_bboxes, all_combined_colors, all_combined_types, 
            all_combined_bboxes, size_multiplier
        )
        
        # Apply size filtering to remove outlier sizes
        size_filtered_bboxes, size_filtered_colors = filter_boxes_by_size(
            filtered_bboxes, filtered_colors, size_ratio_threshold, large_box_ratio
        )
        
        # Filter types to match the filtered boxes
        if len(size_filtered_bboxes) == len(filtered_bboxes):
            # No boxes were filtered out
            size_filtered_types = filtered_types
        else:
            # Need to match filtered boxes with their types
            size_filtered_types = []
            for filtered_bbox, filtered_color in zip(size_filtered_bboxes, size_filtered_colors):
                for i, (orig_bbox, orig_color) in enumerate(zip(filtered_bboxes, filtered_colors)):
                    same_bbox = (filtered_bbox == orig_bbox)
                    same_color = (
                        (filtered_color is None and orig_color is None) or
                        (filtered_color is not None and orig_color is not None and
                        np.all(np.abs(filtered_color.astype(np.float32) - orig_color.astype(np.float32)) < 1e-6))
                    )
                    if same_bbox and same_color:
                        size_filtered_types.append(filtered_types[i])
                        break
        
        # Separate filtered boxes back into their original categories
        filtered_valid_bboxes = []
        filtered_char_bboxes = []
        filtered_char_colors = []
        
        for bbox, color, box_type in zip(size_filtered_bboxes, size_filtered_colors, size_filtered_types):
            if box_type == 'dfs':
                # Reconstruct the valid bbox format (x1, y1, x2, y2, width, height)
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                filtered_valid_bboxes.append((x1, y1, x2, y2, width, height))
            elif box_type == 'color':
                filtered_char_bboxes.append(bbox)
                filtered_char_colors.append(color)
        
        # Update the lists with filtered results
        valid_bboxes = filtered_valid_bboxes
        all_wide_char_bboxes = filtered_char_bboxes
        all_wide_char_colors = filtered_char_colors
        
        # Create final color image with all bounding boxes
        final_color_image = image.copy()
        
        # Draw valid bounding boxes in green
        for valid_bbox in valid_bboxes:
            x1, y1, x2, y2, _, _ = valid_bbox
            cv2.rectangle(final_color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw character bounding boxes from wide boxes using their detected colors
        for char_bbox, char_color in zip(all_wide_char_bboxes, all_wide_char_colors):
            x1, y1, x2, y2 = char_bbox
            # Convert color to tuple for cv2.rectangle
            if char_color is not None and hasattr(char_color, '__iter__'):
                color_tuple = tuple(int(c) for c in char_color)
            else:
                # Use yellow color for boxes with no detected color (original wide boxes)
                color_tuple = (0, 255, 255)  # Yellow in BGR
            cv2.rectangle(final_color_image, (x1, y1), (x2, y2), color_tuple, 2)
        
        # Convert original mask to 3-channel for concatenation
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Create combined image: original on top, mask in middle, smoothened mask with boxes, final color image at bottom
        combined_image = np.vstack([image, mask_3ch, smoothed_mask_with_boxes, final_color_image])
        
        # Create debug folder for combined images
        debug_folder = output_folder / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)
        
        # Create output filename for debug combined image
        debug_filename = f"{image_path.stem}_combined{image_path.suffix}"
        debug_path = debug_folder / debug_filename
        
        # Save the combined image to debug folder
        cv2.imwrite(str(debug_path), combined_image)
        
        # Create segments folder for this image
        image_segments_folder = output_folder / image_path.stem
        image_segments_folder.mkdir(parents=True, exist_ok=True)
        
        # Get image dimensions for padding bounds checking
        img_height, img_width = image.shape[:2]
        
        # Segment and save valid bounding boxes
        segment_count = 0
        for i, valid_bbox in enumerate(valid_bboxes):
            x1, y1, x2, y2, _, _ = valid_bbox
            
            # Apply padding with bounds checking
            x1_padded = max(0, x1 - segment_padding)
            y1_padded = max(0, y1 - segment_padding)
            x2_padded = min(img_width - 1, x2 + segment_padding)
            y2_padded = min(img_height - 1, y2 + segment_padding)
            
            # Extract segment
            segment = image[y1_padded:y2_padded+1, x1_padded:x2_padded+1]
            
            if segment.size > 0:
                # Apply black threshold filtering to make dark pixels white
                processed_segment = apply_black_threshold_to_segment(segment, black_threshold)
                
                segment_filename = f"valid_{i:03d}.png"
                segment_path = image_segments_folder / segment_filename
                cv2.imwrite(str(segment_path), processed_segment)
                segment_count += 1
        
        # Segment and save character bounding boxes from wide boxes with color masking
        for i, (char_bbox, char_color) in enumerate(zip(all_wide_char_bboxes, all_wide_char_colors)):
            x1, y1, x2, y2 = char_bbox
            
            # Apply padding with bounds checking
            x1_padded = max(0, x1 - segment_padding)
            y1_padded = max(0, y1 - segment_padding)
            x2_padded = min(img_width - 1, x2 + segment_padding)
            y2_padded = min(img_height - 1, y2 + segment_padding)
            
            # Extract segment
            segment = image[y1_padded:y2_padded+1, x1_padded:x2_padded+1]
            
            if segment.size > 0:
                # Apply color masking only if a specific color was detected
                if char_color is not None:
                    masked_segment = apply_color_mask_to_segment(segment, char_color, wide_box_color_threshold)
                else:
                    # No specific color detected, use original segment (from wide box fallback)
                    masked_segment = segment
                
                segment_filename = f"char_{i:03d}.png"
                segment_path = image_segments_folder / segment_filename
                cv2.imwrite(str(segment_path), masked_segment)
                segment_count += 1
        
        # Return all bounding boxes for further processing
        return True, valid_bboxes, wide_bboxes, all_wide_char_bboxes, all_wide_char_colors
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False, [], [], [], []

def process_image_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing. Takes a tuple of arguments and unpacks them.
    """
    (image_file, output_path, white_threshold, black_threshold, kernel_size, stride, 
     min_area, width_threshold, segment_padding, color_mask_threshold, 
     wide_box_color_threshold, size_multiplier, size_ratio_threshold, 
     large_box_ratio, color_difference_threshold) = args_tuple
    
    return process_single_image(
        image_file, output_path, white_threshold, black_threshold, kernel_size, 
        stride, min_area, width_threshold, segment_padding, color_mask_threshold, 
        wide_box_color_threshold, size_multiplier, size_ratio_threshold, 
        large_box_ratio, color_difference_threshold
    )

def process_images(input_folder, output_folder, white_threshold=200, black_threshold=50, kernel_size=3, stride=3, min_area=100, width_threshold=1.1, segment_padding=3, color_mask_threshold=70, wide_box_color_threshold=30, size_multiplier=2.0, size_ratio_threshold=0.5, large_box_ratio=2.0, color_difference_threshold=50.0, num_workers=None):
    """
    Process all images in the input folder using multiprocessing.
    
    Args:
        input_folder: Path to input folder containing images
        output_folder: Path to output folder for masks
        white_threshold: White threshold value
        black_threshold: Black threshold value
        kernel_size: Kernel size for smoothening
        stride: Stride for smoothening
        min_area: Minimum area for connected components
        width_threshold: Threshold for filtering wide bounding boxes (width > height * threshold)
        segment_padding: Padding for image segmentation
        color_mask_threshold: Threshold for color masking in character segments
        wide_box_color_threshold: Threshold for color masking when segmenting wide boxes
        size_ratio_threshold: Minimum ratio of box area to median area to keep boxes
        large_box_ratio: Maximum ratio of box area to median area to keep boxes
        color_difference_threshold: Color difference threshold for reclassifying valid boxes as wide boxes
        num_workers: Number of worker processes to use (default: use all CPU cores)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output folder: {output_path}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in '{input_folder}'")
        return
    
    # Sort image files for consistent processing order
    image_files = sorted(image_files)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Found {len(image_files)} image files")
    print(f"Using {num_workers} worker processes")
    print(f"Using thresholds - White: {white_threshold}, Black: {black_threshold}")
    print(f"Using kernel size: {kernel_size}x{kernel_size}")
    print(f"Using minimum area: {min_area}")
    print(f"Using width threshold: {width_threshold}")
    print("-" * 50)
    
    # Prepare arguments for multiprocessing
    args_list = []
    for image_file in image_files:
        args_tuple = (
            image_file, output_path, white_threshold, black_threshold, kernel_size, 
            stride, min_area, width_threshold, segment_padding, color_mask_threshold, 
            wide_box_color_threshold, size_multiplier, size_ratio_threshold, 
            large_box_ratio, color_difference_threshold
        )
        args_list.append(args_tuple)
    
    # Process images using multiprocessing
    successful = 0
    all_valid_bboxes = []
    all_wide_bboxes = []
    all_wide_char_bboxes = []
    
    if num_workers == 1:
        # Process sequentially if only 1 worker specified
        for args_tuple in args_list:
            success, valid_bboxes, wide_bboxes, wide_char_bboxes, wide_char_colors = process_image_wrapper(args_tuple)
            if success:
                successful += 1
                image_file = args_tuple[0]
                # Store bounding boxes with image filename for reference
                all_valid_bboxes.extend([(image_file.name, bbox) for bbox in valid_bboxes])
                all_wide_bboxes.extend([(image_file.name, bbox) for bbox in wide_bboxes])
                all_wide_char_bboxes.extend([(image_file.name, bbox, color) for bbox, color in zip(wide_char_bboxes, wide_char_colors)])
    else:
        # Process in parallel using multiprocessing
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_image_wrapper, args_list)
        
        # Collect results
        for i, (success, valid_bboxes, wide_bboxes, wide_char_bboxes, wide_char_colors) in enumerate(results):
            if success:
                successful += 1
                image_file = image_files[i]
                # Store bounding boxes with image filename for reference
                all_valid_bboxes.extend([(image_file.name, bbox) for bbox in valid_bboxes])
                all_wide_bboxes.extend([(image_file.name, bbox) for bbox in wide_bboxes])
                all_wide_char_bboxes.extend([(image_file.name, bbox, color) for bbox, color in zip(wide_char_bboxes, wide_char_colors)])
    
    print("-" * 50)
    print(f"Processing complete: {successful}/{len(image_files)} images processed successfully")
    print(f"Total valid bounding boxes: {len(all_valid_bboxes)}")
    print(f"Total wide bounding boxes: {len(all_wide_bboxes)}")
    print(f"Total extracted character bounding boxes: {len(all_wide_char_bboxes)}")
    
    return all_valid_bboxes, all_wide_bboxes, all_wide_char_bboxes

def main():
    parser = argparse.ArgumentParser(description="Process images to create color masks")
    parser.add_argument("input_folder", help="Path to input folder containing images")
    parser.add_argument("-o", "--output", default="masks", help="Output folder for masks (default: masks)")
    parser.add_argument("-w", "--white-threshold", type=int, default=250, 
                       help="White threshold value (0-255, default: 250)")
    parser.add_argument("-b", "--black-threshold", type=int, default=5,
                       help="Black threshold value (0-255, default: 5)")
    parser.add_argument("-k", "--kernel-size", type=int, default=3,
                       help="Kernel size for smoothening (default: 3)")
    parser.add_argument("-s", "--stride", type=int, default=3,
                       help="Stride for smoothening (default: 3)")
    parser.add_argument("-m", "--min-area", type=int, default=50,
                       help="Minimum area for connected components (default: 50)")
    parser.add_argument("-t", "--width-threshold", type=float, default=1.1,
                       help="Width threshold for filtering wide bounding boxes (width > height * threshold) (default: 1.1)")
    parser.add_argument("-p", "--segment-padding", type=int, default=3,
                       help="Padding for image segmentation (default: 3 pixels)")
    parser.add_argument("-c", "--color-mask-threshold", type=int, default=70,
                       help="Color similarity threshold for masking character segments (default: 70)")
    parser.add_argument("--wide-box-color-threshold", type=int, default=30,
                       help="Color similarity threshold for masking when segmenting wide boxes (default: 30)")
    parser.add_argument("-mul", type=float, default=2.0,
                       help="Size multiplier for filtering large colored boxes (default: 2.0)")
    parser.add_argument("--size-ratio-threshold", type=float, default=0.5,
                       help="Minimum ratio of box area to median area to keep boxes (default: 0.5)")
    parser.add_argument("--large-box-ratio", type=float, default=2.0,
                       help="Maximum ratio of box area to median area to keep boxes (default: 2.0)")
    parser.add_argument("--color-difference-threshold", type=float, default=50.0,
                       help="Color difference threshold for reclassifying valid boxes as wide boxes (default: 50.0)")
    parser.add_argument("-j", "--workers", type=int, default=None,
                       help="Number of worker processes to use for parallel processing (default: use all CPU cores)")
    
    args = parser.parse_args()
    
    # Validate thresholds
    if not (0 <= args.white_threshold <= 255):
        print("Error: White threshold must be between 0 and 255")
        return
    
    if not (0 <= args.black_threshold <= 255):
        print("Error: Black threshold must be between 0 and 255")
        return
    
    if args.black_threshold >= args.white_threshold:
        print("Error: Black threshold should be less than white threshold")
        return
    
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        print("Error: Kernel size must be a positive odd number")
        return
    
    if args.min_area < 1:
        print("Error: Minimum area must be a positive number")
        return
    
    if args.width_threshold <= 0:
        print("Error: Width threshold must be a positive number")
        return
    
    if args.segment_padding < 0:
        print("Error: Segment padding must be a non-negative number")
        return
    
    if args.mul <= 0:
        print("Error: Size multiplier must be a positive number")
        return
    
    if args.size_ratio_threshold <= 0:
        print("Error: Size ratio threshold must be a positive number")
        return
    
    if args.large_box_ratio <= 0:
        print("Error: Large box ratio must be a positive number")
        return
    
    if args.color_difference_threshold <= 0:
        print("Error: Color difference threshold must be a positive number")
        return
    
    if args.wide_box_color_threshold <= 0:
        print("Error: Wide box color threshold must be a positive number")
        return
    
    if args.workers is not None and args.workers < 1:
        print("Error: Number of workers must be at least 1")
        return
    
    # If workers not specified, use all CPU cores
    num_workers = args.workers if args.workers is not None else cpu_count()
    
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output}")
    print(f"Number of workers: {num_workers}")
    print(f"White threshold: {args.white_threshold}")
    print(f"Black threshold: {args.black_threshold}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Minimum area: {args.min_area}")
    print(f"Width threshold: {args.width_threshold}")
    print(f"Segment padding: {args.segment_padding}")
    print(f"Color mask threshold: {args.color_mask_threshold}")
    print(f"Wide box color threshold: {args.wide_box_color_threshold}")
    print(f"Size multiplier: {args.mul}")
    print(f"Size ratio threshold: {args.size_ratio_threshold}")
    print(f"Large box ratio: {args.large_box_ratio}")
    print(f"Color difference threshold: {args.color_difference_threshold}")
    print("=" * 50)
    
    # Process images
    valid_bboxes, wide_bboxes, wide_char_bboxes = process_images(args.input_folder, args.output, args.white_threshold, args.black_threshold, args.kernel_size, args.stride, args.min_area, args.width_threshold, args.segment_padding, args.color_mask_threshold, args.wide_box_color_threshold, args.mul, args.size_ratio_threshold, args.large_box_ratio, args.color_difference_threshold, num_workers)
    
    # Save hyperparameters to JSON file
    if valid_bboxes is not None and wide_bboxes is not None and wide_char_bboxes is not None:
        # Calculate successful images by counting unique image names
        successful_images = len(set([bbox[0] for bbox in valid_bboxes + wide_bboxes + wide_char_bboxes])) if (valid_bboxes or wide_bboxes or wide_char_bboxes) else 0
        save_unclear_processing_parameters_to_json(
            args, 
            args.output, 
            successful_images,
            len(valid_bboxes) if valid_bboxes else 0,
            len(wide_bboxes) if wide_bboxes else 0,
            len(wide_char_bboxes) if wide_char_bboxes else 0
        )

if __name__ == "__main__":
    # Protect multiprocessing code for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()