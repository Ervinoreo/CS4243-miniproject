import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from scipy.signal import butter, filtfilt
from multiprocessing import Pool, cpu_count, Lock
from functools import partial
import time
from tqdm import tqdm
import threading


# Global lock for thread-safe directory creation
dir_creation_lock = threading.Lock()

def safe_makedirs(directory):
    """
    Thread-safe directory creation.
    
    Args:
        directory: Directory path to create
    """
    with dir_creation_lock:
        os.makedirs(directory, exist_ok=True)


def low_pass_filter(image, cutoff_freq=30):
    """
    Applies a low-pass filter to smooth the image by removing high-frequency components.
    
    Args:
        image: Input grayscale image
        cutoff_freq: Cutoff frequency for the filter
    
    Returns:
        Filtered image
    """
    # Convert to frequency domain
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Create low-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a mask with a circle of radius cutoff_freq
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 <= cutoff_freq**2
    
    # Apply mask and inverse transform
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    filtered_image = np.fft.ifft2(f_ishift)
    filtered_image = np.abs(filtered_image)
    
    return filtered_image.astype(np.uint8)


def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Applies a Gaussian filter for weighted smoothing using a Gaussian kernel.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel
    
    Returns:
        Filtered image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def high_pass_filter(image, cutoff_freq=30):
    """
    Applies a high-pass filter to enhance high-frequency details and edges.
    
    Args:
        image: Input grayscale image
        cutoff_freq: Cutoff frequency for the filter
    
    Returns:
        Filtered image
    """
    # Convert to frequency domain
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Create high-pass filter mask (inverse of low-pass)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a mask with a circle of radius cutoff_freq (inverted)
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 > cutoff_freq**2
    
    # Apply mask and inverse transform
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    filtered_image = np.fft.ifft2(f_ishift)
    filtered_image = np.abs(filtered_image)
    
    return filtered_image.astype(np.uint8)


def laplacian_filter(image, kernel_size=3):
    """
    Applies a Laplacian filter (second-derivative) to highlight rapid intensity changes.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the kernel (1, 3, or 5)
    
    Returns:
        Filtered image
    """
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    
    # Convert back to uint8
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(laplacian)
    
    return laplacian


def sobel_filter(image, dx=1, dy=1, kernel_size=3):
    """
    Applies Sobel filter for gradient-based edge detection with orientation info.
    
    Args:
        image: Input grayscale image
        dx: Order of derivative x
        dy: Order of derivative y
        kernel_size: Size of the Sobel kernel
    
    Returns:
        Filtered image (magnitude of gradients)
    """
    # Calculate gradients in x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=kernel_size)
    
    # Calculate magnitude
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    return sobel_combined


def median_filter(image, kernel_size=5):
    """
    Applies a median filter (non-linear) replacing each pixel with median of neighbors.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the neighborhood kernel
    
    Returns:
        Filtered image
    """
    return cv2.medianBlur(image, kernel_size)


def butterworth_filter(image, cutoff_freq=30, order=2, filter_type='low'):
    """
    Applies Butterworth filter (smooth low/high/band-pass) in frequency domain.
    
    Args:
        image: Input grayscale image
        cutoff_freq: Cutoff frequency for the filter
        order: Order of the Butterworth filter
        filter_type: Type of filter ('low', 'high', 'band')
    
    Returns:
        Filtered image
    """
    # Convert to frequency domain
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create distance matrix
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Create Butterworth filter
    if filter_type == 'low':
        # Low-pass Butterworth filter
        H = 1 / (1 + (distance / cutoff_freq)**(2 * order))
    elif filter_type == 'high':
        # High-pass Butterworth filter
        H = 1 / (1 + (cutoff_freq / (distance + 1e-8))**(2 * order))
    else:  # band-pass (simplified version)
        # Band-pass around the cutoff frequency
        H = distance / (distance + cutoff_freq)
        H = H * (1 / (1 + (distance / (cutoff_freq * 2))**(2 * order)))
    
    # Apply filter and inverse transform
    f_shift_filtered = f_shift * H
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    filtered_image = np.fft.ifft2(f_ishift)
    filtered_image = np.abs(filtered_image)
    
    return filtered_image.astype(np.uint8)


def apply_all_filters(image, params):
    """
    Apply all filters to an image with given parameters.
    
    Args:
        image: Input grayscale image
        params: Dictionary containing filter parameters
    
    Returns:
        Dictionary of filtered images
    """
    filters = {}
    
    try:
        filters['Low-pass'] = low_pass_filter(image, params['low_pass_cutoff'])
    except Exception as e:
        print(f"Error applying low-pass filter: {e}")
        filters['Low-pass'] = image
    
    try:
        filters['Gaussian'] = gaussian_filter(image, params['gaussian_kernel'], params['gaussian_sigma'])
    except Exception as e:
        print(f"Error applying Gaussian filter: {e}")
        filters['Gaussian'] = image
    
    try:
        filters['High-pass'] = high_pass_filter(image, params['high_pass_cutoff'])
    except Exception as e:
        print(f"Error applying high-pass filter: {e}")
        filters['High-pass'] = image
    
    try:
        filters['Laplacian'] = laplacian_filter(image, params['laplacian_kernel'])
    except Exception as e:
        print(f"Error applying Laplacian filter: {e}")
        filters['Laplacian'] = image
    
    try:
        filters['Sobel'] = sobel_filter(image, params['sobel_dx'], params['sobel_dy'], params['sobel_kernel'])
    except Exception as e:
        print(f"Error applying Sobel filter: {e}")
        filters['Sobel'] = image
    
    try:
        filters['Median'] = median_filter(image, params['median_kernel'])
    except Exception as e:
        print(f"Error applying median filter: {e}")
        filters['Median'] = image
    
    try:
        filters['Butterworth'] = butterworth_filter(image, params['butter_cutoff'], params['butter_order'], params['butter_type'])
    except Exception as e:
        print(f"Error applying Butterworth filter: {e}")
        filters['Butterworth'] = image
    
    return filters


def create_combined_visualization(original_image, filtered_images, output_path, image_name):
    """
    Create a combined visualization of original and all filtered images.
    
    Args:
        original_image: Original input image
        filtered_images: Dictionary of filtered images
        output_path: Path to save the combined image
        image_name: Name of the original image
    """
    # Calculate grid size (original + 7 filters = 8 total)
    n_images = len(filtered_images) + 1
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(16, 4 * rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Plot original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot filtered images
    plot_idx = 1
    for filter_name, filtered_img in filtered_images.items():
        row = plot_idx // cols
        col = plot_idx % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(filtered_img, cmap='gray')
        ax.set_title(f'{filter_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_idx += 1
    
    plt.suptitle(f'Local Spatial Filtering Results - {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the combined image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_image(args):
    """
    Process a single image with filters. This function is designed for multiprocessing.
    Each image knows its destination folder and handles thread-safe saving.
    
    Args:
        args: Tuple containing (image_path, output_path, output_folder, params)
    
    Returns:
        Tuple of (success, image_file, subfolder_name, error_message)
    """
    image_path, output_path, output_folder, params = args
    image_file = os.path.basename(image_path)
    subfolder_name = os.path.basename(os.path.dirname(image_path))
    
    try:
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return (False, image_file, subfolder_name, "Could not read image")
        
        # Apply all filters
        filtered_images = apply_all_filters(image, params)
        
        # Ensure output directory exists (thread-safe)
        safe_makedirs(output_folder)
        
        # Create combined visualization and save
        create_combined_visualization(image, filtered_images, output_path, image_file)
        
        return (True, image_file, subfolder_name, None)
        
    except Exception as e:
        return (False, image_file, subfolder_name, str(e))


def process_images(input_folder, params, num_processes=None):
    """
    Process all images in the input folder and its subfolders using parallel processing.
    Each image is processed individually in parallel.
    
    Args:
        input_folder: Path to input folder containing subfolders with images
        params: Dictionary containing filter parameters
        num_processes: Number of processes to use (None for CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()  # Use all CPU cores
    
    print(f"Using {num_processes} parallel processes for individual images")
    
    # Create output folder
    base_name = os.path.basename(input_folder.rstrip('/\\'))
    output_folder = f"{input_folder}_local_spatial_debug"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing images from: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Collect all image files with their destination paths
    image_tasks = []
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(output_folder, subfolder_name)
            
            # Get all PNG image files in the subfolder
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.png')]
            
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)
                output_name = f"{os.path.splitext(image_file)[0]}_filtered.png"
                output_path = os.path.join(output_subfolder, output_name)
                
                # Add to task list: (image_path, output_path, output_subfolder, params)
                image_tasks.append((image_path, output_path, output_subfolder, params))
    
    if not image_tasks:
        print("No images found to process.")
        return
    
    start_time = time.time()
    total_processed = 0
    total_images = len(image_tasks)
    subfolder_stats = {}  # Track stats per subfolder
    
    print(f"Processing {total_images} images in parallel...")
    
    # Process all images in parallel
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress of image completion
        with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
            for result in pool.imap(process_single_image, image_tasks):
                success, image_file, subfolder_name, error_msg = result
                
                # Update subfolder statistics
                if subfolder_name not in subfolder_stats:
                    subfolder_stats[subfolder_name] = {'processed': 0, 'total': 0, 'errors': []}
                
                subfolder_stats[subfolder_name]['total'] += 1
                
                if success:
                    total_processed += 1
                    subfolder_stats[subfolder_name]['processed'] += 1
                else:
                    subfolder_stats[subfolder_name]['errors'].append(f"{image_file}: {error_msg}")
                    print(f"    Error processing {image_file} in {subfolder_name}: {error_msg}")
                
                # Update progress bar with current subfolder info
                pbar.set_postfix({
                    'Current': subfolder_name[:10] + ('...' if len(subfolder_name) > 10 else ''),
                    'Success': f"{total_processed}/{total_images}"
                })
                pbar.update(1)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total images processed: {total_processed}/{total_images}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/total_processed:.3f} seconds" if total_processed > 0 else "N/A")
    
    # Print detailed subfolder statistics
    print(f"\nSubfolder Statistics:")
    for subfolder_name, stats in subfolder_stats.items():
        print(f"  {subfolder_name}: {stats['processed']}/{stats['total']} images processed")
        if stats['errors']:
            print(f"    Errors: {len(stats['errors'])}")
    
    print(f"Results saved to: {output_folder}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Apply local spatial filters to images with parallel processing')
    
    # Input/output arguments
    parser.add_argument('input_folder', help='Input folder containing subfolders with PNG images')
    
    # Parallel processing parameters
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of parallel processes to use (default: CPU count, each image processed in parallel)')
    
    # Low-pass filter parameters
    parser.add_argument('--low_pass_cutoff', type=float, default=30.0,
                       help='Cutoff frequency for low-pass filter (default: 30.0)')
    
    # Gaussian filter parameters
    parser.add_argument('--gaussian_kernel', type=int, default=5,
                       help='Kernel size for Gaussian filter (must be odd, default: 5)')
    parser.add_argument('--gaussian_sigma', type=float, default=1.0,
                       help='Sigma for Gaussian filter (default: 1.0)')
    
    # High-pass filter parameters
    parser.add_argument('--high_pass_cutoff', type=float, default=30.0,
                       help='Cutoff frequency for high-pass filter (default: 30.0)')
    
    # Laplacian filter parameters
    parser.add_argument('--laplacian_kernel', type=int, default=3,
                       help='Kernel size for Laplacian filter (1, 3, or 5, default: 3)')
    
    # Sobel filter parameters
    parser.add_argument('--sobel_dx', type=int, default=1,
                       help='Order of derivative x for Sobel filter (default: 1)')
    parser.add_argument('--sobel_dy', type=int, default=1,
                       help='Order of derivative y for Sobel filter (default: 1)')
    parser.add_argument('--sobel_kernel', type=int, default=3,
                       help='Kernel size for Sobel filter (default: 3)')
    
    # Median filter parameters
    parser.add_argument('--median_kernel', type=int, default=5,
                       help='Kernel size for median filter (must be odd, default: 5)')
    
    # Butterworth filter parameters
    parser.add_argument('--butter_cutoff', type=float, default=30.0,
                       help='Cutoff frequency for Butterworth filter (default: 30.0)')
    parser.add_argument('--butter_order', type=int, default=2,
                       help='Order for Butterworth filter (default: 2)')
    parser.add_argument('--butter_type', type=str, default='low', choices=['low', 'high', 'band'],
                       help='Type of Butterworth filter (default: low)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return
    
    # Validate parameters
    if args.gaussian_kernel % 2 == 0:
        print("Error: Gaussian kernel size must be odd.")
        return
    
    if args.median_kernel % 2 == 0:
        print("Error: Median kernel size must be odd.")
        return
    
    if args.laplacian_kernel not in [1, 3, 5]:
        print("Error: Laplacian kernel size must be 1, 3, or 5.")
        return
    
    # Validate number of processes
    if args.num_processes is not None and args.num_processes < 1:
        print("Error: Number of processes must be at least 1.")
        return
    
    # Create parameter dictionary
    params = {
        'low_pass_cutoff': args.low_pass_cutoff,
        'gaussian_kernel': args.gaussian_kernel,
        'gaussian_sigma': args.gaussian_sigma,
        'high_pass_cutoff': args.high_pass_cutoff,
        'laplacian_kernel': args.laplacian_kernel,
        'sobel_dx': args.sobel_dx,
        'sobel_dy': args.sobel_dy,
        'sobel_kernel': args.sobel_kernel,
        'median_kernel': args.median_kernel,
        'butter_cutoff': args.butter_cutoff,
        'butter_order': args.butter_order,
        'butter_type': args.butter_type
    }
    
    print("Local Spatial Feature Extraction (Parallelized by Individual Images)")
    print("=" * 50)
    print(f"Input folder: {args.input_folder}")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Processes to use: {args.num_processes if args.num_processes else cpu_count()}")
    print("Strategy: Each image processed in parallel with thread-safe folder creation")
    print("Filter parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    print("=" * 50)
    
    # Process images with parallel processing
    process_images(args.input_folder, params, args.num_processes)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()
