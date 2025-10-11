import os
import json
import numpy as np
from datetime import datetime


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1, point2: Tuples of (x, y) coordinates
    
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def segmentation_save_parameters_to_json(args, output_dir, total_images, total_components):
    """
    Save processing parameters to a JSON file in the output directory.
    
    Args:
        args: Parsed command line arguments
        output_dir: Output directory path
        total_images: Total number of images processed
        total_components: Total number of components found
    """
    parameters = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_images_processed": total_images,
            "total_components_found": total_components
        },
        "input_parameters": {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "color_threshold": args.color_threshold,
            "padding": args.padding,
            "distance_threshold": args.distance_threshold,
            "area_ratio_threshold": args.area_ratio_threshold,
            "white_threshold": args.white_threshold,
            "black_threshold": args.black_threshold,
            "size_ratio_threshold": args.size_ratio_threshold,
            "large_box_ratio": args.large_box_ratio,
            "extensions": args.extensions
        },
        "algorithm_parameters": {
            "dbscan_eps": 15,
            "dbscan_min_samples": 30,
            "dbscan_threshold": 0.005,
            "min_component_size": 20,
            "connectivity": "8-connected"
        }
    }
    
    # Save to JSON file
    json_path = os.path.join(output_dir, "processing_parameters.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"Parameters saved to: {json_path}")


def save_unclear_processing_parameters_to_json(args, output_dir, total_images, total_valid_boxes, total_wide_boxes, total_char_boxes):
    """
    Save unclear image processing parameters to a JSON file in the debug directory.
    
    Args:
        args: Parsed command line arguments
        output_dir: Output directory path
        total_images: Total number of images processed successfully
        total_valid_boxes: Total number of valid bounding boxes found
        total_wide_boxes: Total number of wide bounding boxes found
        total_char_boxes: Total number of character bounding boxes extracted
    """
    parameters = {
        "processing_info": {
            "script": "process_unclear_images.py",
            "timestamp": datetime.now().isoformat(),
            "total_images_processed": total_images,
            "total_valid_boxes_found": total_valid_boxes,
            "total_wide_boxes_found": total_wide_boxes,
            "total_character_boxes_extracted": total_char_boxes
        },
        "input_parameters": {
            "input_folder": args.input_folder,
            "output_folder": args.output,
            "white_threshold": args.white_threshold,
            "black_threshold": args.black_threshold,
            "kernel_size": args.kernel_size,
            "stride": args.stride,
            "min_area": args.min_area,
            "width_threshold": args.width_threshold,
            "segment_padding": args.segment_padding,
            "color_mask_threshold": args.color_mask_threshold,
            "size_multiplier": args.mul,
            "size_ratio_threshold": args.size_ratio_threshold,
            "large_box_ratio": args.large_box_ratio
        },
        "dfs_algorithm_parameters": {
            "connectivity": "8-connected (including diagonals)",
            "directions": [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        },
        "wide_box_processing_parameters": {
            "dbscan_eps": 15,
            "dbscan_min_samples": 50,
            "dbscan_threshold": 0.005,
            "color_threshold": 40,
            "padding": 3,
            "distance_threshold": 30,
            "area_ratio_threshold": 0.2,
            "density_ratio_threshold": 0.25
        },
        "processing_pipeline": [
            "1. Create color mask (white/black threshold filtering)",
            "2. Apply smoothing mask (kernel averaging)",
            "3. Find connected components using DFS",
            "4. Separate valid and wide bounding boxes",
            "5. Process wide boxes with DBSCAN color detection",
            "6. Apply UFDS merging within wide boxes",
            "7. Apply distance-based merging within wide boxes", 
            "8. Apply nested component merging within wide boxes",
            "9. Apply pixel density filtering within wide boxes",
            "10. Combine all bounding boxes",
            "11. Filter large colored boxes",
            "12. Apply size-based filtering",
            "13. Generate debug visualizations",
            "14. Segment and save character images"
        ]
    }
    
    # Save to JSON file in debug directory
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    json_path = os.path.join(debug_dir, "hyperparameters.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"Hyperparameters saved to: {json_path}")