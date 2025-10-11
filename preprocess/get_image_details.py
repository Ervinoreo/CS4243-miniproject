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


def process_folder(folder_path: str, output_file: str = None, 
                  min_stroke_width: float = 2.5, min_laplacian_var: float = 1000.0,
                  filter_images_flag: bool = False, output_folder: str = None) -> None:
    """
    Process all images in a folder and save statistics to a text file.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_file (str): Path to output text file (optional)
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
    
    # Generate output filename if not provided
    if output_file is None:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        suffix = "_filtered" if filter_images_flag else ""
        output_file = f"image_statistics_{folder_name}{suffix}.txt"
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Image Statistics Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Folder: {folder_path}\n")
        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Generated on: {os.path.basename(__file__)}\n")
        
        if filter_images_flag:
            f.write(f"Filtering Enabled:\n")
            f.write(f"  - Minimum Stroke Width: {min_stroke_width} pixels\n")
            f.write(f"  - Minimum Laplacian Variance: {min_laplacian_var}\n")
            f.write(f"  - Valid Images: {filter_stats['valid_count']}\n")
            f.write(f"  - Filtered Out: {filter_stats['filtered_count']}\n")
            if output_folder:
                f.write(f"  - Output Folder: {output_folder}\n")
        f.write("\n")
        
        # Write individual image statistics
        for filename, stats in results.items():
            if 'error' in stats:
                f.write(f"Image: {filename}\n")
                f.write(f"Error: {stats['error']}\n")
                f.write("-" * 50 + "\n")
                continue
                
            f.write(f"Image: {stats['filename']}\n")
            f.write(f"Dimensions: {stats['width']} x {stats['height']}\n")
            f.write(f"Laplacian Variance: {stats['laplacian_variance']:.2f}\n")
            f.write(f"Clarity Level: {stats['clarity_level']}\n")
            f.write(f"Is Clear: {'Yes' if stats['is_clear'] else 'No'}\n")
            f.write(f"Mean Intensity: {stats['mean_intensity']:.2f}\n")
            f.write(f"Standard Deviation: {stats['std_intensity']:.2f}\n")
            f.write(f"Mean Stroke Width: {stats['mean_stroke_width']:.2f} pixels\n")
            f.write(f"Median Stroke Width: {stats['median_stroke_width']:.2f} pixels\n")
            f.write(f"Stroke Width Std Dev: {stats['std_stroke_width']:.2f} pixels\n")
            f.write(f"Character Thickness: {stats['thickness_level']}\n")
            f.write(f"Text Regions Found: {stats['num_text_regions']}\n")
            f.write("-" * 50 + "\n")
        
        # Write summary statistics
        valid_results = [stats for stats in results.values() if 'error' not in stats]
        if valid_results:
            f.write("\nSUMMARY STATISTICS\n")
            f.write("=" * 50 + "\n")
            
            # Clarity distribution
            clarity_levels = {}
            thickness_levels = {}
            laplacian_values = []
            stroke_widths = []
            
            for stats in valid_results:
                clarity_level = stats['clarity_level']
                thickness_level = stats['thickness_level']
                
                clarity_levels[clarity_level] = clarity_levels.get(clarity_level, 0) + 1
                thickness_levels[thickness_level] = thickness_levels.get(thickness_level, 0) + 1
                
                laplacian_values.append(stats['laplacian_variance'])
                if stats['mean_stroke_width'] > 0:
                    stroke_widths.append(stats['mean_stroke_width'])
            
            f.write("Clarity Distribution:\n")
            for level, count in clarity_levels.items():
                f.write(f"  {level}: {count} images\n")
            
            f.write("\nThickness Distribution:\n")
            for level, count in thickness_levels.items():
                f.write(f"  {level}: {count} images\n")
            
            f.write(f"\nOverall Statistics:\n")
            f.write(f"  Average Laplacian Variance: {np.mean(laplacian_values):.2f}\n")
            f.write(f"  Std Dev Laplacian Variance: {np.std(laplacian_values):.2f}\n")
            
            if stroke_widths:
                f.write(f"  Average Stroke Width: {np.mean(stroke_widths):.2f} pixels\n")
                f.write(f"  Std Dev Stroke Width: {np.std(stroke_widths):.2f} pixels\n")
            
            clear_count = sum(1 for stats in valid_results if stats['is_clear'])
            f.write(f"  Clear Images: {clear_count}/{len(valid_results)} ({clear_count/len(valid_results)*100:.1f}%)\n")
        
        # Write filtering results if filtering was applied
        if filter_images_flag and filter_stats:
            f.write("\nFILTERING RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Images: {filter_stats['total_processed']}\n")
            f.write(f"Valid Images: {filter_stats['valid_count']}\n")
            f.write(f"Filtered Out: {filter_stats['filtered_count']}\n\n")
            
            if filter_stats['valid_count'] > 0:
                f.write("VALID IMAGES:\n")
                f.write("-" * 30 + "\n")
                for img_info in filter_stats['valid_images']:
                    f.write(f"{img_info['filename']}: ")
                    f.write(f"stroke={img_info['stroke_width']:.2f}, ")
                    f.write(f"clarity={img_info['laplacian_var']:.2f}\n")
                f.write("\n")
            
            if filter_stats['filtered_count'] > 0:
                f.write("FILTERED OUT IMAGES:\n")
                f.write("-" * 30 + "\n")
                for img_info in filter_stats['filtered_out']:
                    f.write(f"{img_info['filename']}: {img_info['reason']}\n")
    
    print(f"Results saved to: {output_file}")
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
    parser.add_argument('second_image', nargs='?', help='Second image path (for comparison mode)')
    parser.add_argument('-o', '--output', help='Output file path for statistics')
    parser.add_argument('--filter', action='store_true', help='Enable image filtering')
    parser.add_argument('--output-folder', help='Output folder for filtered images')
    parser.add_argument('--min-stroke-width', type=float, default=2.5, 
                       help='Minimum stroke width threshold (default: 2.5)')
    parser.add_argument('--min-clarity', type=float, default=1000.0,
                       help='Minimum Laplacian variance threshold (default: 1000.0)')
    
    args = parser.parse_args()
    
    # Check if input is a folder or file
    if os.path.isdir(args.input):
        # Folder mode
        if args.second_image:
            print("Warning: Second image argument ignored in folder mode")
        
        process_folder(
            folder_path=args.input,
            output_file=args.output,
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
        
        # Compare the two images
        if len(results) == 2:
            image_names = list(results.keys())
            img1_stats = results[image_names[0]]
            img2_stats = results[image_names[1]]
            
            if 'laplacian_variance' in img1_stats and 'laplacian_variance' in img2_stats:
                print("Comparison:")
                if img1_stats['laplacian_variance'] > img2_stats['laplacian_variance']:
                    print(f"{img1_stats['filename']} is clearer than {img2_stats['filename']}")
                elif img2_stats['laplacian_variance'] > img1_stats['laplacian_variance']:
                    print(f"{img2_stats['filename']} is clearer than {img1_stats['filename']}")
                else:
                    print("Both images have similar clarity levels")
    
    else:
        print("Error: Invalid input. Please provide either:")
        print("  - A folder path for batch processing")
        print("  - Two image paths for comparison")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
