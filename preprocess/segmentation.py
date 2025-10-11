import os
import argparse
import cv2
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
from filter_unclear_images import get_image_statistics
from process_unclear_images import process_single_image as process_image_segments


def process_single_image(image_path: str, output_folder: str, min_stroke_width: float, min_clarity: float) -> Dict[str, Any]:
    """
    Process a single image to determine if it's clear and suitable for processing.
    
    Args:
        image_path (str): Path to the input image
        output_folder (str): Output folder path
        min_stroke_width (float): Minimum stroke width threshold
        min_clarity (float): Minimum clarity (Laplacian variance) threshold
        
    Returns:
        Dict[str, Any]: Processing results for the image
    """
    try:
        # Get image statistics using the function from filter_unclear_images.py
        stats = get_image_statistics(image_path)
        
        # Determine if image meets criteria
        is_suitable = True
        rejection_reasons = []
        
        # Check stroke width (only if text was detected)
        if stats['mean_stroke_width'] > 0 and stats['mean_stroke_width'] < min_stroke_width:
            is_suitable = False
            rejection_reasons.append(f"Thin stroke (width: {stats['mean_stroke_width']:.2f} < {min_stroke_width})")
        
        # Check clarity
        if stats['laplacian_variance'] < min_clarity:
            is_suitable = False
            rejection_reasons.append(f"Not clear (variance: {stats['laplacian_variance']:.2f} < {min_clarity})")
        
        result = {
            'filename': os.path.basename(image_path),
            'image_path': image_path,
            'is_suitable': is_suitable,
            'rejection_reasons': rejection_reasons,
            'stats': stats,
            'processed': True
        }
        
        return result
        
    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'image_path': image_path,
            'is_suitable': False,
            'rejection_reasons': [f"Processing error: {str(e)}"],
            'stats': None,
            'processed': False,
            'error': str(e)
        }


def get_image_files(input_folder: str) -> List[str]:
    """
    Get all image files from the input folder.
    
    Args:
        input_folder (str): Path to the input folder
        
    Returns:
        List[str]: List of image file paths
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_folder, filename))
    
    return image_files


def process_images_parallel(input_folder: str, output_folder: str, 
                          min_stroke_width: float, min_clarity: float,
                          max_workers: int = None) -> Dict[str, Any]:
    """
    Process multiple images in parallel to determine clarity and suitability.
    
    Args:
        input_folder (str): Path to input folder containing images
        output_folder (str): Path to output folder
        min_stroke_width (float): Minimum stroke width threshold
        min_clarity (float): Minimum clarity threshold
        max_workers (int): Maximum number of worker processes
        
    Returns:
        Dict[str, Any]: Processing results and statistics
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(input_folder)
    
    if not image_files:
        print(f"No image files found in '{input_folder}'")
        return {
            'results': [],
            'suitable_count': 0,
            'unsuitable_count': 0,
            'error_count': 0,
            'total_count': 0
        }
    
    print(f"Found {len(image_files)} image(s) in '{input_folder}'")
    print(f"Processing images with {max_workers or mp.cpu_count()} workers...")
    
    # Process images in parallel
    results = []
    suitable_count = 0
    unsuitable_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, image_path, output_folder, 
                          min_stroke_width, min_clarity): image_path 
            for image_path in image_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            try:
                result = future.result()
                results.append(result)
                
                if result['processed']:
                    if result['is_suitable']:
                        suitable_count += 1
                        print(f"✓ {result['filename']} - Suitable")
                    else:
                        unsuitable_count += 1
                        print(f"✗ {result['filename']} - Unsuitable: {'; '.join(result['rejection_reasons'])}")
                else:
                    error_count += 1
                    print(f"! {result['filename']} - Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                image_path = future_to_image[future]
                filename = os.path.basename(image_path)
                error_count += 1
                print(f"! {filename} - Processing failed: {str(e)}")
                results.append({
                    'filename': filename,
                    'image_path': image_path,
                    'is_suitable': False,
                    'rejection_reasons': [f"Processing failed: {str(e)}"],
                    'stats': None,
                    'processed': False,
                    'error': str(e)
                })
    
    return {
        'results': results,
        'suitable_count': suitable_count,
        'unsuitable_count': unsuitable_count,
        'error_count': error_count,
        'total_count': len(image_files)
    }


def save_results_to_file(results: Dict[str, Any], output_folder: str) -> None:
    """
    Save processing results to a text file.
    
    Args:
        results (Dict[str, Any]): Processing results
        output_folder (str): Output folder path
    """
    results_file = os.path.join(output_folder, 'segmentation_analysis.txt')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Image Segmentation Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total images processed: {results['total_count']}\n")
        f.write(f"Suitable images: {results['suitable_count']}\n")
        f.write(f"Unsuitable images: {results['unsuitable_count']}\n")
        f.write(f"Processing errors: {results['error_count']}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results['results']:
            f.write(f"\nImage: {result['filename']}\n")
            f.write(f"Status: {'Suitable' if result['is_suitable'] else 'Unsuitable'}\n")
            
            if result['processed'] and result['stats']:
                stats = result['stats']
                f.write(f"Dimensions: {stats['width']} x {stats['height']}\n")
                f.write(f"Laplacian Variance: {stats['laplacian_variance']:.2f}\n")
                f.write(f"Clarity Level: {stats['clarity_level']}\n")
                f.write(f"Mean Stroke Width: {stats['mean_stroke_width']:.2f} pixels\n")
                f.write(f"Character Thickness: {stats['thickness_level']}\n")
                f.write(f"Text Regions Found: {stats['num_text_regions']}\n")
            
            if result['rejection_reasons']:
                f.write(f"Rejection Reasons: {'; '.join(result['rejection_reasons'])}\n")
            
            f.write("-" * 30 + "\n")
    
    print(f"\nDetailed results saved to: {results_file}")


def save_hyperparameters_to_json(args, output_folder: str, processing_results: Dict[str, Any]) -> None:
    """
    Save all hyperparameters to a JSON file in the debug folder.
    
    Args:
        args: Command line arguments
        output_folder (str): Output folder path
        processing_results (Dict[str, Any]): Processing results with statistics
    """
    debug_folder = os.path.join(output_folder, 'debug')
    os.makedirs(debug_folder, exist_ok=True)
    
    hyperparameters = {
        "processing_info": {
            "total_images": processing_results['total_count'],
            "suitable_images": processing_results['suitable_count'],
            "unsuitable_images": processing_results['unsuitable_count'],
            "error_count": processing_results['error_count']
        },
        "image_classification_parameters": {
            "min_stroke_width": args.min_stroke_width,
            "min_clarity": args.min_clarity,
            "workers": args.workers
        },
        "unclear_image_processing": {
            "enabled": args.process_unclear,
            "white_threshold": args.white_threshold,
            "black_threshold": args.black_threshold,
            "kernel_size": args.kernel_size,
            "stride": args.stride,
            "min_area": args.min_area,
            "width_threshold": args.width_threshold,
            "segment_padding": args.segment_padding,
            "color_mask_threshold": args.color_mask_threshold,
            "size_multiplier": args.size_multiplier
        },
        "clear_image_processing": {
            "enabled": args.process_clear,
            "color_threshold": args.clear_color_threshold,
            "padding": args.clear_padding,
            "distance_threshold": args.clear_distance_threshold,
            "area_ratio_threshold": args.clear_area_ratio_threshold,
            "denoise_black_threshold": args.clear_denoise_black_threshold,
            "size_ratio_threshold": args.clear_size_ratio_threshold,
            "large_box_ratio": args.clear_large_box_ratio,
            "density_ratio_threshold": args.clear_density_ratio_threshold,
            "detected_color_threshold": args.clear_detected_color_threshold,
            "debug_mode": args.clear_debug and not args.combine
        },
        "general_settings": {
            "combine_mode": args.combine,
            "input_folder": args.input_folder,
            "output_folder": args.output_folder
        }
    }
    
    json_file = os.path.join(debug_folder, 'hyperparameters.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=2, ensure_ascii=False)
    
    print(f"Hyperparameters saved to: {json_file}")


def main():
    """Main function to handle command line arguments and process images."""
    parser = argparse.ArgumentParser(
        description='Process images for segmentation by analyzing clarity and stroke width'
    )
    
    parser.add_argument('input_folder', 
                       help='Input folder containing images to process')
    parser.add_argument('output_folder',
                       help='Output folder for results and processed images')
    parser.add_argument('-msw', '--min-stroke-width', 
                       type=float, default=2.5,
                       help='Minimum stroke width threshold (default: 2.5)')
    parser.add_argument('-clar', '--min-clarity',
                       type=float, default=1500.0,
                       help='Minimum clarity (Laplacian variance) threshold (default: 1500.0)')
    parser.add_argument('-w', '--workers',
                       type=int, default=None,
                       help='Number of worker processes (default: number of CPU cores)')
    
    # Image processing parameters for unclear images
    parser.add_argument('--process-unclear', action='store_true',
                       help='Process unclear images through image segmentation pipeline')
    parser.add_argument('--white-threshold', type=int, default=250,
                       help='White threshold for image processing (default: 250)')
    parser.add_argument('--black-threshold', type=int, default=5,
                       help='Black threshold for image processing (default: 5)')
    parser.add_argument('--kernel-size', type=int, default=3,
                       help='Kernel size for smoothening (default: 3)')
    parser.add_argument('--stride', type=int, default=3,
                       help='Stride for smoothening (default: 3)')
    parser.add_argument('--min-area', type=int, default=50,
                       help='Minimum area for connected components (default: 50)')
    parser.add_argument('--width-threshold', type=float, default=1.1,
                       help='Width threshold for filtering wide bounding boxes (default: 1.1)')
    parser.add_argument('--segment-padding', type=int, default=3,
                       help='Padding for image segmentation (default: 3)')
    parser.add_argument('--color-mask-threshold', type=int, default=20,
                       help='Color mask threshold for character segments (default: 20)')
    parser.add_argument('--size-multiplier', type=float, default=2.0,
                       help='Size multiplier for filtering large colored boxes (default: 2.0)')
    
    # Clear image processing parameters
    parser.add_argument('--process-clear', action='store_true',
                       help='Process clear images through clear image segmentation pipeline')
    parser.add_argument('--clear-color-threshold', type=float, default=40.0,
                       help='Color similarity threshold for clear images (default: 40.0)')
    parser.add_argument('--clear-padding', type=int, default=3,
                       help='Padding around segmented characters for clear images (default: 3)')
    parser.add_argument('--clear-distance-threshold', type=float, default=30.0,
                       help='Maximum distance between centroids to merge components (default: 30.0)')
    parser.add_argument('--clear-area-ratio-threshold', type=float, default=0.2,
                       help='Maximum ratio of small/large area to merge nested components (default: 0.2)')
    parser.add_argument('--clear-denoise-black-threshold', type=int, default=10,
                       help='Threshold for denoising extremely dark pixels (default: 10)')
    parser.add_argument('--clear-size-ratio-threshold', type=float, default=0.5,
                       help='Minimum ratio of box area to median area to keep boxes (default: 0.5)')
    parser.add_argument('--clear-large-box-ratio', type=float, default=2.0,
                       help='Maximum ratio of box area to median area to keep boxes (default: 2.0)')
    parser.add_argument('--clear-density-ratio-threshold', type=float, default=0.25,
                       help='Minimum ratio of box pixel density to median density (default: 0.25)')
    parser.add_argument('--clear-debug', action='store_true',
                       help='Enable debug mode for clear image processing')
    parser.add_argument('--clear-detected-color-threshold', type=float, default=30.0,
                       help='Threshold for detecting the target color in clear image character segmentation (default: 30.0)')
    
    # Combine mode parameter
    parser.add_argument('--combine', action='store_true',
                       help='Combine clear and unclear processing outputs into unified folders')
    
    args = parser.parse_args()
    
    try:
        # Process images
        results = process_images_parallel(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            min_stroke_width=args.min_stroke_width,
            min_clarity=args.min_clarity,
            max_workers=args.workers
        )
        
        # Print summary
        print(f"\nProcessing Complete!")
        print(f"Total images: {results['total_count']}")
        print(f"Suitable images: {results['suitable_count']}")
        print(f"Unsuitable images: {results['unsuitable_count']}")
        print(f"Processing errors: {results['error_count']}")
        
        # Save detailed results (skip in combine mode)
        if not args.combine:
            save_results_to_file(results, args.output_folder)
        
        # Process unclear images through image processing pipeline if requested
        if args.process_unclear and results['unsuitable_count'] > 0:
            print(f"\nProcessing {results['unsuitable_count']} unclear images through image segmentation pipeline...")
            
            # Create subfolder for unclear image processing
            if args.combine:
                unclear_output_folder = args.output_folder  # Use main output folder in combine mode
            else:
                unclear_output_folder = os.path.join(args.output_folder, "unclear_processed")
                os.makedirs(unclear_output_folder, exist_ok=True)
            
            # Process each unclear image
            unclear_processed = 0
            unclear_errors = 0
            
            for result in results['results']:
                if not result['is_suitable'] and result['processed']:
                    try:
                        from pathlib import Path
                        image_path = Path(result['image_path'])
                        
                        print(f"Processing unclear image: {result['filename']}")
                        
                        # Process the image through the segmentation pipeline
                        success, valid_bboxes, wide_bboxes, wide_char_bboxes, wide_char_colors = process_image_segments(
                            image_path=image_path,
                            output_folder=Path(unclear_output_folder),
                            white_threshold=args.white_threshold,
                            black_threshold=args.black_threshold,
                            kernel_size=args.kernel_size,
                            stride=args.stride,
                            min_area=args.min_area,
                            width_threshold=args.width_threshold,
                            segment_padding=args.segment_padding,
                            color_mask_threshold=args.color_mask_threshold,
                            size_multiplier=args.size_multiplier
                        )
                        
                        if success:
                            unclear_processed += 1
                            print(f"  ✓ Processed {result['filename']} - Found {len(valid_bboxes)} valid boxes, {len(wide_char_bboxes)} character boxes")
                        else:
                            unclear_errors += 1
                            print(f"  ✗ Failed to process {result['filename']}")
                            
                    except Exception as e:
                        unclear_errors += 1
                        print(f"  ✗ Error processing {result['filename']}: {str(e)}")
            
            print(f"\nUnclear image processing complete:")
            print(f"  Successfully processed: {unclear_processed}")
            print(f"  Processing errors: {unclear_errors}")
            print(f"  Output folder: {unclear_output_folder}")
        
        # Process clear images through clear image segmentation pipeline if requested
        if args.process_clear and results['suitable_count'] > 0:
            print(f"\nProcessing {results['suitable_count']} clear images through clear image segmentation pipeline...")
            
            # Create subfolder for clear image processing
            if args.combine:
                clear_output_folder = args.output_folder  # Use main output folder in combine mode
            else:
                clear_output_folder = os.path.join(args.output_folder, "clear_processed")
                os.makedirs(clear_output_folder, exist_ok=True)
            
            # Import clear image processing function
            from clear_image_segmentation import process_single_image as process_clear_image
            from clear_image_segmentation import create_processing_parameters
            
            # Create processing parameters for clear images
            class ClearArgs:
                def __init__(self, args):
                    self.color_threshold = args.clear_color_threshold
                    self.padding = args.clear_padding
                    self.distance_threshold = args.clear_distance_threshold
                    self.area_ratio_threshold = args.clear_area_ratio_threshold
                    self.white_threshold = args.white_threshold  # Use same white threshold
                    self.black_threshold = args.black_threshold  # Use same black threshold
                    self.denoise_black_threshold = args.clear_denoise_black_threshold
                    self.size_ratio_threshold = args.clear_size_ratio_threshold
                    self.large_box_ratio = args.clear_large_box_ratio
                    self.density_ratio_threshold = args.clear_density_ratio_threshold
                    self.detected_color_threshold = args.clear_detected_color_threshold
                    # Force debug mode off in combine mode
                    self.debug = args.clear_debug and not args.combine
            
            clear_args = ClearArgs(args)
            clear_processing_params = create_processing_parameters(clear_args)
            
            # Collect clear images for parallel processing
            clear_images = [result for result in results['results'] if result['is_suitable'] and result['processed']]
            
            # Process clear images in parallel
            clear_processed = 0
            clear_errors = 0 
            total_clear_components = 0
            
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Submit all clear image processing tasks
                future_to_result = {
                    executor.submit(process_clear_image, result['image_path'], clear_output_folder, clear_processing_params): result 
                    for result in clear_images
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_result):
                    result = future_to_result[future]
                    try:
                        processed_path, components_found, status = future.result()
                        
                        if status == "Success":
                            clear_processed += 1
                            total_clear_components += components_found
                            print(f"  ✓ Processed {result['filename']} - Found {components_found} components")
                        else:
                            clear_errors += 1
                            print(f"  ✗ Failed to process {result['filename']}: {status}")
                            
                    except Exception as e:
                        clear_errors += 1
                        print(f"  ✗ Error processing {result['filename']}: {str(e)}")
            
            print(f"\nClear image processing complete:")
            print(f"  Successfully processed: {clear_processed}")
            print(f"  Processing errors: {clear_errors}")
            print(f"  Total components found: {total_clear_components}")
            if not args.combine:
                print(f"  Output folder: {clear_output_folder}")
        
        # Print combine mode summary and save hyperparameters
        if args.combine and (args.process_clear or args.process_unclear):
            # Save hyperparameters to JSON file in debug folder
            save_hyperparameters_to_json(args, args.output_folder, results)
            
            print(f"\n{'='*60}")
            print("COMBINE MODE SUMMARY")
            print(f"{'='*60}")
            print(f"All processed images and debug files are combined in:")
            print(f"  Output folder: {args.output_folder}")
            
            if args.process_clear and args.process_unclear:
                total_processed = clear_processed + unclear_processed if 'clear_processed' in locals() and 'unclear_processed' in locals() else 0
                total_errors = clear_errors + unclear_errors if 'clear_errors' in locals() and 'unclear_errors' in locals() else 0
                print(f"  Total images processed: {total_processed}")
                print(f"  Total processing errors: {total_errors}")
                if 'total_clear_components' in locals():
                    print(f"  Total components found (clear): {total_clear_components}")
            
            print(f"  Debug images: Combined in {os.path.join(args.output_folder, 'debug')}")
            print(f"  Individual character images: Organized by image name")
            print(f"  Hyperparameters: Saved in {os.path.join(args.output_folder, 'debug', 'hyperparameters.json')}")
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
