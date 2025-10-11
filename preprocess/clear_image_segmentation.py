"""
Main Image Segmentation Module

This module orchestrates the complete image segmentation pipeline for character extraction.
It coordinates between different modules to provide a clean, maintainable solution.
"""

import os
import cv2
import numpy as np
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial

# Import from our modular components
from utils import segmentation_save_parameters_to_json
from color_analysis import extract_main_colors_dbscan, create_color_mask
from clear_connected_components import find_connected_components
from bounding_box import (
    get_bounding_box_from_component,
    merge_nested_components,
    merge_components_with_ufds,
    merge_nearby_components,
    filter_boxes_by_size,
    filter_boxes_by_pixel_density,
    draw_bounding_boxes
)
from clear_character_segmentation import segment_character


def count_pixels_in_bounding_box(image, bbox, color, color_threshold, white_threshold):
    """
    Count pixels within a bounding box that match the specified color criteria.
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box tuple (x1, y1, x2, y2)
        color: Target color (BGR format)
        color_threshold: Color similarity threshold
        white_threshold: Threshold for white pixels
    
    Returns:
        Dictionary with pixel counts and statistics
    """
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    
    # Create color mask for the ROI
    color_mask = create_color_mask(roi, color, color_threshold)
    
    # Count different types of pixels
    total_pixels = roi.shape[0] * roi.shape[1]
    color_pixels = np.sum(color_mask)
    
    # Count white pixels (pixels above white threshold in all channels)
    white_mask = np.all(roi >= white_threshold, axis=2)
    white_pixels = np.sum(white_mask)
    
    # Count non-white, non-color pixels
    other_pixels = total_pixels - color_pixels - white_pixels
    
    return {
        'total_pixels': int(total_pixels),
        'color_pixels': int(color_pixels),
        'white_pixels': int(white_pixels),
        'other_pixels': int(other_pixels),
        'bbox_width': x2 - x1,
        'bbox_height': y2 - y1,
        'color_ratio': float(color_pixels / total_pixels) if total_pixels > 0 else 0.0,
        'white_ratio': float(white_pixels / total_pixels) if total_pixels > 0 else 0.0
    }


def process_single_image(image_path, output_dir, processing_params):
    """
    Process a single image and create character segmentations.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory path
        processing_params: Dictionary containing all processing parameters
    
    Returns:
        Tuple of (image_path, components_found, success_status)
    """
    # Extract parameters for cleaner code
    color_threshold = processing_params['color_threshold']
    padding = processing_params['padding']
    distance_threshold = processing_params['distance_threshold']
    area_ratio_threshold = processing_params['area_ratio_threshold']
    white_threshold = processing_params['white_threshold']
    black_threshold = processing_params['black_threshold']
    size_ratio_threshold = processing_params['size_ratio_threshold']
    large_box_ratio = processing_params['large_box_ratio']
    density_ratio_threshold = processing_params['density_ratio_threshold']
    
    # Read and validate image
    image = cv2.imread(image_path)
    if image is None:
        return (image_path, 0, f"Error: Could not read image")
    
    # Step 1: Preprocessing - denoise the image
    # denoised_image = denoise_image(image, denoise_black_threshold)
    denoised_image = image.copy()
    
    # Step 2: Color Analysis - extract main colors
    main_colors = extract_main_colors_dbscan(
        denoised_image, 
        eps=15, 
        min_samples=50, 
        threshold=0.005,
        white_threshold=white_threshold, 
        black_threshold=black_threshold
    )
    
    if not main_colors:
        return (image_path, 0, "No main colors found")
    
    # Step 3: Component Detection - find connected components for each color
    all_bounding_boxes = []
    all_colors = []
    
    for color in main_colors:
        # Create color mask
        color_mask = create_color_mask(denoised_image, color, color_threshold)
        
        # Find connected components
        components = find_connected_components(color_mask, min_component_size=1)
        
        # Convert components to bounding boxes
        for component in components:
            bbox = get_bounding_box_from_component(
                component, 
                padding=padding, 
                image_shape=denoised_image.shape[:2]
            )
            if bbox is not None:
                all_bounding_boxes.append(bbox)
                all_colors.append(color)
    
    if not all_bounding_boxes:
        return (image_path, 0, "No valid connected components found")
    
    # Step 4: Component Merging Pipeline
    # 4a. Merge nearby components with similar colors using ufds
    ufds_merged_boxes, ufds_merged_colors = merge_components_with_ufds(
        all_bounding_boxes, all_colors, color_threshold, distance_threshold
    )

    # 4b. Merge nearby components with similar colors using distance-based method
    nearby_merged_boxes, nearby_merged_colors = merge_nearby_components(
        ufds_merged_boxes, ufds_merged_colors, color_threshold, distance_threshold
    )
    
    # 4c. Merge nested components with similar colors
    nested_merged_boxes, nested_merged_colors = merge_nested_components(
        nearby_merged_boxes, nearby_merged_colors, color_threshold, area_ratio_threshold
    )
    
    if not nested_merged_boxes:
        return (image_path, 0, "No valid merged components found")
    
    # 4c. Filter out outlier sizes
    size_filtered_boxes, size_filtered_colors = filter_boxes_by_size(
        nested_merged_boxes, nested_merged_colors, size_ratio_threshold, large_box_ratio
    )
    
    if not size_filtered_boxes:
        return (image_path, 0, "No components remaining after size filtering")
    
    # 4d. Filter out low pixel density boxes
    final_merged_boxes, final_merged_colors = filter_boxes_by_pixel_density(
        denoised_image, size_filtered_boxes, size_filtered_colors, density_ratio_threshold, white_threshold
    )
    
    if not final_merged_boxes:
        return (image_path, 0, "No components remaining after pixel density filtering")
    
    # Step 5: Output Generation
    _save_processing_results(
        image_path, 
        output_dir, 
        denoised_image, 
        final_merged_boxes, 
        final_merged_colors,
        all_bounding_boxes,
        all_colors,
        main_colors,
        color_threshold,
        black_threshold,
        debug_mode=processing_params.get('debug_mode', False),
        white_threshold=white_threshold,
        detected_color_threshold=processing_params.get('detected_color_threshold', 30)
    )
    
    return (image_path, len(final_merged_boxes), "Success")


def _save_processing_results(image_path, output_dir, denoised_image, bounding_boxes, colors, original_boxes, original_colors, main_colors, color_threshold, black_threshold, debug_mode=False, white_threshold=245, detected_color_threshold=30):
    """
    Save the processing results including debug images and segmented characters.
    
    Args:
        image_path: Path to the original image
        output_dir: Output directory
        denoised_image: Preprocessed image
        bounding_boxes: Final bounding boxes
        colors: Corresponding colors
        original_boxes: Original bounding boxes before merging
        original_colors: Original colors before merging
        main_colors: List of main colors detected
        color_threshold: Color threshold for segmentation
        black_threshold: Black threshold for segmentation
        debug_mode: Whether to save additional debug files
        white_threshold: Threshold for white pixels
    """
    image_name = Path(image_path).stem
    debug_dir = os.path.join(output_dir, "debug")
    image_dir = os.path.join(output_dir, image_name)
    
    # Always create these directories
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Create additional debug directories only if debug mode is enabled
    if debug_mode:
        before_merging_dir = os.path.join(output_dir, "before_merging")
        color_masks_dir = os.path.join(output_dir, "color_masks")
        os.makedirs(before_merging_dir, exist_ok=True)
        os.makedirs(color_masks_dir, exist_ok=True)
    
    # Save debug files only if debug mode is enabled
    if debug_mode:
        # Save color masks for each detected color
        for i, color in enumerate(main_colors):
            color_mask = create_color_mask(denoised_image, color, color_threshold)
            # Convert boolean mask to uint8 if needed
            if color_mask.dtype == bool:
                color_mask = color_mask.astype(np.uint8) * 255
            # Convert binary mask to 3-channel for visualization
            color_mask_vis = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            color_mask_path = os.path.join(color_masks_dir, f"{image_name}_color{i}_mask_BGR{color[0]}-{color[1]}-{color[2]}.png")
            cv2.imwrite(color_mask_path, color_mask_vis)
        
        # Save before merging debug image with original bounding boxes
        before_merging_image = draw_bounding_boxes(denoised_image, original_boxes, original_colors)
        before_merging_path = os.path.join(before_merging_dir, f"{image_name}_before_merging.png")
        cv2.imwrite(before_merging_path, before_merging_image)
    
    # Always save debug image with final bounding boxes
    debug_image = draw_bounding_boxes(denoised_image, bounding_boxes, colors)
    debug_path = os.path.join(debug_dir, f"{image_name}_debug.png")
    cv2.imwrite(debug_path, debug_image)
    
    # Segment and save each character
    for i, (bbox, color) in enumerate(zip(bounding_boxes, colors)):
        char_image = segment_character(
            denoised_image, 
            bbox, 
            color,  # Pass the detected color
            color_threshold, 
            detected_color_threshold=detected_color_threshold,
            padding=0,  # No additional padding as it's already applied
            black_threshold=black_threshold
        )
        if char_image is not None:
            char_path = os.path.join(image_dir, f"{image_name}_char{i}.png")
            cv2.imwrite(char_path, char_image)


def create_processing_parameters(args):
    """
    Create a parameters dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Dictionary containing all processing parameters
    """
    return {
        'color_threshold': args.color_threshold,
        'padding': args.padding,
        'distance_threshold': args.distance_threshold,
        'area_ratio_threshold': args.area_ratio_threshold,
        'white_threshold': args.white_threshold,
        'black_threshold': args.black_threshold,
        'denoise_black_threshold': args.denoise_black_threshold,
        'size_ratio_threshold': args.size_ratio_threshold,
        'large_box_ratio': args.large_box_ratio,
        'density_ratio_threshold': args.density_ratio_threshold,
        'debug_mode': args.debug,
        'detected_color_threshold': getattr(args, 'detected_color_threshold', 30)
    }


def get_image_files(input_dir, extensions):
    """
    Get list of image files from input directory.
    
    Args:
        input_dir: Input directory path
        extensions: List of file extensions to look for
    
    Returns:
        List of unique image file paths
    """
    image_files = []
    for ext in extensions:
        # Add both lowercase and uppercase extensions to avoid duplicates
        pattern_lower = f"*{ext.lower()}"
        pattern_upper = f"*{ext.upper()}"
        image_files.extend(Path(input_dir).glob(pattern_lower))
        image_files.extend(Path(input_dir).glob(pattern_upper))
    
    # Remove duplicates by converting to set and back to list
    return list(set(image_files))


def process_batch_parallel(image_files, output_dir, processing_params, num_workers=None, batch_size=100):
    """
    Process multiple images in parallel using multiprocessing.
    
    Args:
        image_files: List of image file paths
        output_dir: Output directory path
        processing_params: Dictionary containing all processing parameters
        num_workers: Number of worker processes (default: CPU count)
        batch_size: Number of files to process in each progress report
    
    Returns:
        Tuple of (total_components, successful_images, failed_images)
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(image_files))
    
    print(f"Using {num_workers} worker processes")
    
    # Create a partial function with fixed arguments
    process_func = partial(process_single_image, 
                          output_dir=output_dir, 
                          processing_params=processing_params)
    
    total_components = 0
    successful_images = 0
    failed_images = []
    processed_count = 0
    
    start_time = time.time()
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(process_func, str(image_path)): image_path 
            for image_path in image_files
        }
        
        # Process completed jobs
        for future in as_completed(future_to_path):
            image_path = future_to_path[future]
            processed_count += 1
            
            try:
                result_path, components_found, status = future.result()
                
                if status == "Success":
                    total_components += components_found
                    successful_images += 1
                    
                    # Progress reporting
                    if processed_count % batch_size == 0 or processed_count == len(image_files):
                        elapsed_time = time.time() - start_time
                        images_per_second = processed_count / elapsed_time
                        eta_seconds = (len(image_files) - processed_count) / images_per_second if images_per_second > 0 else 0
                        eta_minutes = eta_seconds / 60
                        
                        print(f"Progress: {processed_count}/{len(image_files)} "
                              f"({processed_count/len(image_files)*100:.1f}%) - "
                              f"Speed: {images_per_second:.2f} imgs/sec - "
                              f"ETA: {eta_minutes:.1f} min - "
                              f"Components found: {components_found}")
                else:
                    failed_images.append((str(image_path), status))
                    print(f"Failed {image_path.name}: {status}")
                    
            except Exception as e:
                failed_images.append((str(image_path), f"Exception: {str(e)}"))
                print(f"Exception processing {image_path.name}: {str(e)}")
    
    return total_components, successful_images, failed_images


def process_batch_sequential(image_files, output_dir, processing_params, batch_size=100):
    """
    Process multiple images sequentially (fallback option).
    
    Args:
        image_files: List of image file paths
        output_dir: Output directory path
        processing_params: Dictionary containing all processing parameters
        batch_size: Number of files to process in each progress report
    
    Returns:
        Tuple of (total_components, successful_images, failed_images)
    """
    total_components = 0
    successful_images = 0
    failed_images = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        result_path, components_found, status = process_single_image(
            str(image_path), output_dir, processing_params
        )
        
        if status == "Success":
            total_components += components_found
            successful_images += 1
        else:
            failed_images.append((str(image_path), status))
            print(f"Failed {image_path.name}: {status}")
        
        # Progress reporting
        if (i + 1) % batch_size == 0 or (i + 1) == len(image_files):
            elapsed_time = time.time() - start_time
            images_per_second = (i + 1) / elapsed_time
            eta_seconds = (len(image_files) - (i + 1)) / images_per_second if images_per_second > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f"Progress: {i+1}/{len(image_files)} "
                  f"({(i+1)/len(image_files)*100:.1f}%) - "
                  f"Speed: {images_per_second:.2f} imgs/sec - "
                  f"ETA: {eta_minutes:.1f} min - "
                  f"Components found: {components_found}")
    
    return total_components, successful_images, failed_images


def main():
    """Main function to handle command line arguments and orchestrate processing."""
    parser = argparse.ArgumentParser(description="Image Character Segmentation")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for segmented images")
    parser.add_argument("--color_threshold", type=float, default=40.0,
                       help="Color similarity threshold (default: 40.0)")
    parser.add_argument("--padding", type=int, default=3,
                       help="Padding around segmented characters in pixels (default: 3)")
    parser.add_argument("--distance_threshold", type=float, default=30.0,
                       help="Maximum distance between centroids to merge components (default: 30.0)")
    parser.add_argument("--area_ratio_threshold", type=float, default=0.2,
                       help="Maximum ratio of small/large area to merge nested components (default: 0.2)")
    parser.add_argument("--white_threshold", type=int, default=245,
                       help="Threshold for filtering white pixels - lower values capture lighter characters (default: 245)")
    parser.add_argument("--black_threshold", type=int, default=10,
                       help="Threshold for filtering black pixels - higher values capture darker characters (default: 10)")
    parser.add_argument("--denoise_black_threshold", type=int, default=10,
                       help="Threshold for denoising extremely dark pixels during preprocessing (default: 10)")
    parser.add_argument("--size_ratio_threshold", type=float, default=0.5,
                       help="Minimum ratio of box area to median area to keep boxes (default: 0.5)")
    parser.add_argument("--large_box_ratio", type=float, default=2.0,
                       help="Maximum ratio of box area to median area to keep boxes (default: 2.0)")
    parser.add_argument("--density_ratio_threshold", type=float, default=0.25,
                       help="Minimum ratio of box pixel density to median density to keep boxes (default: 0.25)")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"],
                       help="Image file extensions to process")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--sequential", action="store_true",
                       help="Process images sequentially instead of in parallel")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Progress reporting interval (default: 100)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode - saves additional debug files and statistics")
    parser.add_argument("--detected_color_threshold", type=float, default=30.0,
                       help="Threshold for detecting the target color in character segmentation (default: 30.0)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    image_files = get_image_files(args.input_dir, args.extensions)
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    # Create processing parameters
    processing_params = create_processing_parameters(args)
    
    # Display processing information
    print(f"Found {len(image_files)} images to process")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Color threshold: {args.color_threshold}")
    print(f"Padding: {args.padding} pixels")
    print(f"UFDS distance threshold: {args.distance_threshold} pixels")
    print(f"Area ratio threshold: {args.area_ratio_threshold}")
    print(f"White threshold: {args.white_threshold}")
    print(f"Black threshold: {args.black_threshold}")
    print(f"Denoise black threshold: {args.denoise_black_threshold}")
    print(f"Size ratio threshold: {args.size_ratio_threshold}")
    print(f"Density ratio threshold: {args.density_ratio_threshold}")
    print(f"Output directory: {args.output_dir}")
    
    # Process images (parallel or sequential)
    start_time = time.time()
    
    if args.sequential:
        print("Processing images sequentially...")
        total_components, successful_images, failed_images = process_batch_sequential(
            image_files, args.output_dir, processing_params, args.batch_size
        )
    else:
        print("Processing images in parallel...")
        total_components, successful_images, failed_images = process_batch_parallel(
            image_files, args.output_dir, processing_params, args.workers, args.batch_size
        )
    
    # Calculate processing statistics
    end_time = time.time()
    total_time = end_time - start_time
    images_per_second = len(image_files) / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful: {successful_images}")
    print(f"Failed: {len(failed_images)}")
    print(f"Total components found: {total_components}")
    print(f"Processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average speed: {images_per_second:.2f} images/second")
    
    # Display failed images if any
    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for img_path, error in failed_images[:10]:  # Show first 10 failures
            print(f"  - {Path(img_path).name}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print("="*60)
    
    # Save parameters to JSON file
    segmentation_save_parameters_to_json(args, args.output_dir, successful_images, total_components)


if __name__ == "__main__":
    # This guard is necessary for multiprocessing on Windows
    mp.freeze_support()
    main()