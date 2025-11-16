import cv2
import numpy as np
import os
import sys
import argparse
import shutil
from typing import Dict, Any
from scipy import ndimage


def calculate_laplacian_variance(image_path: str) -> float:
    """
    Calculate the Laplacian variance of an image to measure its sharpness/clarity.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        float: Laplacian variance (higher values indicate sharper images)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate variance of Laplacian
    variance = laplacian.var()
    
    return variance


def calculate_stroke_width_transform(gray_image: np.ndarray) -> Dict[str, float]:
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


def get_image_statistics(image_path: str) -> Dict[str, Any]:
    """
    Get comprehensive image statistics including clarity metrics.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Dict[str, Any]: Dictionary containing image statistics
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (blur detection)
    laplacian_var = calculate_laplacian_variance(image_path)
    
    # Calculate stroke width statistics
    stroke_stats = calculate_stroke_width_transform(gray)
    
    # Additional image statistics
    height, width = gray.shape
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Determine if image is clear based on Laplacian variance
    # Threshold can be adjusted based on your specific use case
    clarity_threshold = 100  # This is a common threshold, but may need tuning
    is_clear = laplacian_var > clarity_threshold
    
    # Classify clarity level
    if laplacian_var > 500:
        clarity_level = "Very Sharp"
    elif laplacian_var > 200:
        clarity_level = "Sharp"
    elif laplacian_var > 100:
        clarity_level = "Moderately Sharp"
    elif laplacian_var > 50:
        clarity_level = "Slightly Blurry"
    else:
        clarity_level = "Very Blurry"
    
    return {
        'filename': os.path.basename(image_path),
        'width': width,
        'height': height,
        'laplacian_variance': laplacian_var,
        'is_clear': is_clear,
        'clarity_level': clarity_level,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'clarity_threshold': clarity_threshold,
        'mean_stroke_width': stroke_stats['mean_stroke_width'],
        'std_stroke_width': stroke_stats['std_stroke_width'],
        'median_stroke_width': stroke_stats['median_stroke_width'],
        'thickness_level': stroke_stats['thickness_level'],
        'num_text_regions': stroke_stats['num_text_regions']
    }


def analyze_multiple_images(image_paths: list) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple images and return their statistics.
    
    Args:
        image_paths (list): List of image file paths
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with filename as key and statistics as value
    """
    results = {}
    
    for image_path in image_paths:
        try:
            stats = get_image_statistics(image_path)
            results[stats['filename']] = stats
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results[os.path.basename(image_path)] = {'error': str(e)}
    
    return results


def print_image_statistics(stats: Dict[str, Any]) -> None:
    """
    Print image statistics in a readable format.
    
    Args:
        stats (Dict[str, Any]): Image statistics dictionary
    """
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return
        
    print(f"Image: {stats['filename']}")
    print(f"Dimensions: {stats['width']} x {stats['height']}")
    print(f"Laplacian Variance: {stats['laplacian_variance']:.2f}")
    print(f"Clarity Level: {stats['clarity_level']}")
    print(f"Is Clear: {'Yes' if stats['is_clear'] else 'No'}")
    print(f"Mean Intensity: {stats['mean_intensity']:.2f}")
    print(f"Standard Deviation: {stats['std_intensity']:.2f}")
    print(f"Mean Stroke Width: {stats['mean_stroke_width']:.2f} pixels")
    print(f"Median Stroke Width: {stats['median_stroke_width']:.2f} pixels")
    print(f"Stroke Width Std Dev: {stats['std_stroke_width']:.2f} pixels")
    print(f"Character Thickness: {stats['thickness_level']}")
    print(f"Text Regions Found: {stats['num_text_regions']}")
    print("-" * 50)


def filter_images(results: Dict[str, Dict[str, Any]], folder_path: str, 
                 min_stroke_width: float = 2.5, min_laplacian_var: float = 1000.0,
                 output_folder: str = None) -> Dict[str, Any]:
    """
    Filter images based on stroke width and clarity criteria.
    
    Args:
        results (Dict): Image analysis results
        folder_path (str): Original folder path
        min_stroke_width (float): Minimum mean stroke width threshold
        min_laplacian_var (float): Minimum Laplacian variance threshold
        output_folder (str): Optional output folder to copy filtered images
        
    Returns:
        Dict: Filtering statistics
    """
    valid_images = []
    filtered_out = []
    
    for filename, stats in results.items():
        if 'error' in stats:
            filtered_out.append({
                'filename': filename,
                'reason': f"Error: {stats['error']}"
            })
            continue
            
        # Check filtering criteria
        reasons = []
        
        # Check stroke width (only if text was detected)
        if stats['mean_stroke_width'] > 0 and stats['mean_stroke_width'] < min_stroke_width:
            reasons.append(f"Thin stroke (width: {stats['mean_stroke_width']:.2f} < {min_stroke_width})")
        
        # Check clarity
        if stats['laplacian_variance'] < min_laplacian_var:
            reasons.append(f"Not clear (variance: {stats['laplacian_variance']:.2f} < {min_laplacian_var})")
        
        if reasons:
            filtered_out.append({
                'filename': filename,
                'reason': "; ".join(reasons),
                'stroke_width': stats['mean_stroke_width'],
                'laplacian_var': stats['laplacian_variance']
            })
        else:
            valid_images.append({
                'filename': filename,
                'stroke_width': stats['mean_stroke_width'],
                'laplacian_var': stats['laplacian_variance'],
                'stats': stats
            })
    
    # Copy filtered out images to output folder if specified
    if output_folder and filtered_out:
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nCopying {len(filtered_out)} filtered out images to '{output_folder}'...")
        
        for img_info in filtered_out:
            src_path = os.path.join(folder_path, img_info['filename'])
            dst_path = os.path.join(output_folder, img_info['filename'])
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {img_info['filename']}: {e}")
    
    return {
        'valid_images': valid_images,
        'filtered_out': filtered_out,
        'total_processed': len(results),
        'valid_count': len(valid_images),
        'filtered_count': len(filtered_out)
    }


def process_folder(folder_path: str,
                  min_stroke_width: float = 2.5, min_laplacian_var: float = 1000.0,
                  filter_images_flag: bool = False, output_folder: str = None) -> None:
    """
    Process all images in a folder and save statistics to a text file.
    
    Args:
        folder_path (str): Path to the folder containing images
        min_stroke_width (float): Minimum mean stroke width threshold
        min_laplacian_var (float): Minimum Laplacian variance threshold
        filter_images_flag (bool): Whether to filter images
        output_folder (str): Output folder for filtered images
    """
    # Get all image files in the folder
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, filename))
    
    if not image_paths:
        print(f"No image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_paths)} image(s) in '{folder_path}'")
    print("Processing images...")
    
    # Analyze all images
    results = analyze_multiple_images(image_paths)
    
    # Filter images if requested
    filter_stats = None
    if filter_images_flag:
        filter_stats = filter_images(results, folder_path, min_stroke_width, 
                                   min_laplacian_var, output_folder)
        valid_results = filter_stats['valid_images']

    print(f"Processed {len(valid_results)} images successfully")
    
    if filter_images_flag and filter_stats:
        print(f"\nFiltering Results:")
        print(f"  Valid images: {filter_stats['valid_count']}")
        print(f"  Filtered out: {filter_stats['filtered_count']}")
        if output_folder:
            print(f"  Valid images copied to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Analyze image statistics and optionally filter images')
    parser.add_argument('input', help='Input folder path or first image path')
    parser.add_argument('--filter', action='store_true', help='Enable image filtering')
    parser.add_argument('--output-folder', help='Output folder for filtered images')
    parser.add_argument('--min-stroke-width', type=float, default=2.5, 
                       help='Minimum stroke width threshold (default: 2.5)')
    parser.add_argument('--min-clarity', type=float, default=1000.0,
                       help='Minimum Laplacian variance threshold (default: 1000.0)')
    
    args = parser.parse_args()
    
    # Check if input is a folder or file
    if os.path.isdir(args.input):
        
        process_folder(
            folder_path=args.input,
            min_stroke_width=args.min_stroke_width,
            min_laplacian_var=args.min_clarity,
            filter_images_flag=args.filter,
            output_folder=args.output_folder
        )
        
    elif args.second_image and os.path.isfile(args.input) and os.path.isfile(args.second_image):
        # Two images comparison mode
        if args.filter:
            print("Warning: Filtering not available in comparison mode")
        
        example_images = [args.input, args.second_image]
        
        print("Image Clarity Analysis using Laplacian Variance")
        print("=" * 60)
        
        # Analyze the example images
        results = analyze_multiple_images(example_images)
        
        for filename, stats in results.items():
            print_image_statistics(stats)
    
    
    else:
        print("Error: Invalid input. Please provide either:")
        print("  - A folder path for batch processing")
        print("  - Two image paths for comparison")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
