import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import threading
import time

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


def glcm_features(image, distances=[1], angles=[0, 45, 90, 135]):
    """
    Gray-Level Co-occurrence Matrix (GLCM) features.
    Measures how often pixel pairs with specific values occur at a distance.
    
    Args:
        image: Input grayscale image
        distances: List of pixel distances
        angles: List of angles in degrees
    
    Returns:
        GLCM feature image
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert angles to radians
        angles_rad = [np.deg2rad(angle) for angle in angles]
        
        # Compute GLCM
        glcm = graycomatrix(image, distances=distances, angles=angles_rad, 
                           levels=256, symmetric=True, normed=True)
        
        # Extract features
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Create feature image (simple visualization)
        feature_img = np.ones_like(image, dtype=np.float32)
        feature_img = feature_img * contrast * 50  # Scale for visualization
        feature_img = np.clip(feature_img, 0, 255)
        
        return feature_img.astype(np.uint8)
        
    except ImportError:
        # Fallback: Simple co-occurrence calculation
        h, w = image.shape
        cooc = np.zeros((256, 256), dtype=np.float32)
        
        # Calculate co-occurrence for horizontal neighbors
        for i in range(h):
            for j in range(w-1):
                cooc[image[i,j], image[i,j+1]] += 1
        
        # Normalize
        cooc = cooc / (cooc.sum() + 1e-8)
        
        # Simple contrast measure
        contrast = 0
        for i in range(256):
            for j in range(256):
                contrast += cooc[i,j] * (i-j)**2
        
        # Create visualization
        feature_img = np.ones_like(image, dtype=np.float32) * contrast * 0.1
        feature_img = np.clip(feature_img, 0, 255)
        
        return feature_img.astype(np.uint8)


def lbp_features(image, radius=3, n_points=24):
    """
    Local Binary Patterns (LBP).
    Encodes pixel neighborhood into binary pattern.
    
    Args:
        image: Input grayscale image
        radius: Radius of circle around pixel
        n_points: Number of circularly symmetric neighbor set points
    
    Returns:
        LBP feature image
    """
    try:
        from skimage.feature import local_binary_pattern
        
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Normalize to 0-255
        lbp_norm = ((lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8) * 255)
        
        return lbp_norm.astype(np.uint8)
        
    except ImportError:
        # Simple fallback LBP implementation
        h, w = image.shape
        lbp_image = np.zeros_like(image)
        
        # Simple 3x3 LBP
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                # 8 neighbors in clockwise order
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code += 2**k
                
                lbp_image[i, j] = code
        
        # Normalize
        lbp_norm = ((lbp_image - lbp_image.min()) / (lbp_image.max() - lbp_image.min() + 1e-8) * 255)
        
        return lbp_norm.astype(np.uint8)


def texton_features(image, n_textons=16, patch_size=5):
    """
    Texton analysis.
    Fundamental repeating texture elements identified via clustering.
    
    Args:
        image: Input grayscale image
        n_textons: Number of textons to extract
        patch_size: Size of patches for texton analysis
    
    Returns:
        Texton feature image
    """
    h, w = image.shape
    
    # Extract patches
    patches = []
    positions = []
    
    for i in range(0, h - patch_size + 1, patch_size // 2):
        for j in range(0, w - patch_size + 1, patch_size // 2):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
            positions.append((i, j))
    
    if len(patches) == 0:
        return image
    
    patches = np.array(patches)
    
    # Cluster patches to find textons
    try:
        kmeans = KMeans(n_clusters=min(n_textons, len(patches)), random_state=42, n_init=10)
        texton_labels = kmeans.fit_predict(patches)
        
        # Create texton map
        texton_image = np.zeros_like(image)
        
        for (i, j), label in zip(positions, texton_labels):
            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            texton_image[i:end_i, j:end_j] = label * (255 // n_textons)
        
        return texton_image.astype(np.uint8)
        
    except:
        # Fallback: simple patch variance
        texton_image = np.zeros_like(image)
        
        for (i, j), patch in zip(positions, patches):
            variance = np.var(patch)
            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            texton_image[i:end_i, j:end_j] = min(255, variance * 10)
        
        return texton_image.astype(np.uint8)


def autocorrelation_features(image, max_displacement=10):
    """
    Autocorrelation analysis.
    Measures spatial repetition of pixel patterns.
    
    Args:
        image: Input grayscale image
        max_displacement: Maximum displacement for autocorrelation
    
    Returns:
        Autocorrelation feature image
    """
    h, w = image.shape
    
    # Calculate autocorrelation for different displacements
    autocorr_sum = np.zeros_like(image, dtype=np.float32)
    
    for dx in range(-max_displacement, max_displacement + 1):
        for dy in range(-max_displacement, max_displacement + 1):
            if dx == 0 and dy == 0:
                continue
            
            # Calculate valid regions
            y1, y2 = max(0, dy), min(h, h + dy)
            x1, x2 = max(0, dx), min(w, w + dx)
            
            y1_shift, y2_shift = max(0, -dy), min(h, h - dy)
            x1_shift, x2_shift = max(0, -dx), min(w, w - dx)
            
            if y2 > y1 and x2 > x1 and y2_shift > y1_shift and x2_shift > x1_shift:
                # Calculate correlation
                region1 = image[y1:y2, x1:x2].astype(np.float32)
                region2 = image[y1_shift:y2_shift, x1_shift:x2_shift].astype(np.float32)
                
                correlation = region1 * region2
                
                # Add to autocorrelation sum
                autocorr_sum[y1:y2, x1:x2] += correlation
    
    # Normalize
    autocorr_sum = autocorr_sum / (2 * max_displacement + 1)**2
    autocorr_sum = np.clip(autocorr_sum, 0, 255)
    
    return autocorr_sum.astype(np.uint8)


def pca_texture_features(image, patch_size=8, n_components=3):
    """
    PCA-based texture analysis.
    Reduces dimensionality by projecting onto principal axes.
    
    Args:
        image: Input grayscale image
        patch_size: Size of patches for PCA analysis
        n_components: Number of principal components
    
    Returns:
        PCA texture feature image
    """
    h, w = image.shape
    
    # Extract patches
    patches = []
    positions = []
    
    for i in range(0, h - patch_size + 1, patch_size // 2):
        for j in range(0, w - patch_size + 1, patch_size // 2):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
            positions.append((i, j))
    
    if len(patches) == 0:
        return image
    
    patches = np.array(patches)
    
    # Apply PCA
    try:
        pca = PCA(n_components=min(n_components, patches.shape[1]))
        pca_patches = pca.fit_transform(patches)
        
        # Create feature image using first principal component
        pca_image = np.zeros_like(image, dtype=np.float32)
        
        for (i, j), pca_vals in zip(positions, pca_patches):
            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            pca_image[i:end_i, j:end_j] = abs(pca_vals[0])
        
        # Normalize
        pca_image = 255 * pca_image / (pca_image.max() + 1e-8)
        
        return pca_image.astype(np.uint8)
        
    except:
        # Fallback: variance-based feature
        var_image = np.zeros_like(image, dtype=np.float32)
        
        for (i, j), patch in zip(positions, patches):
            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)
            var_image[i:end_i, j:end_j] = np.var(patch)
        
        # Normalize
        var_image = 255 * var_image / (var_image.max() + 1e-8)
        
        return var_image.astype(np.uint8)


def msmd_features(image, scales=[1, 2, 4], directions=[0, 45, 90, 135]):
    """
    Multi-Scale Multi-Directional (MSMD) texture analysis.
    Analyzes texture across scales and orientations.
    
    Args:
        image: Input grayscale image
        scales: List of scales for analysis
        directions: List of directions in degrees
    
    Returns:
        MSMD feature image
    """
    h, w = image.shape
    msmd_response = np.zeros_like(image, dtype=np.float32)
    
    for scale in scales:
        for direction in directions:
            # Create directional filter
            kernel_size = max(3, scale * 3)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create oriented kernel
            x = np.arange(kernel_size) - kernel_size // 2
            y = np.arange(kernel_size) - kernel_size // 2
            X, Y = np.meshgrid(x, y)
            
            # Rotate coordinates
            theta = np.deg2rad(direction)
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            
            # Create directional derivative kernel
            sigma = scale
            kernel = X_rot * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)
            
            # Apply filter
            try:
                from scipy.signal import convolve2d
                filtered = convolve2d(image, kernel, mode='same', boundary='symm')
                msmd_response += np.abs(filtered)
            except:
                # Simple convolution fallback
                filtered = ndimage.convolve(image.astype(np.float32), kernel, mode='reflect')
                msmd_response += np.abs(filtered)
    
    # Normalize
    msmd_response = msmd_response / (len(scales) * len(directions))
    msmd_response = np.clip(msmd_response, 0, 255)
    
    return msmd_response.astype(np.uint8)


def apply_all_texture_methods(image, params):
    """
    Apply all texture analysis methods to an image with given parameters.
    
    Args:
        image: Input grayscale image
        params: Dictionary containing method parameters
    
    Returns:
        Dictionary of texture feature images
    """
    features = {}
    
    try:
        features['GLCM'] = glcm_features(image, params['glcm_distances'], params['glcm_angles'])
    except Exception as e:
        print(f"Error applying GLCM: {e}")
        features['GLCM'] = image
    
    try:
        features['LBP'] = lbp_features(image, params['lbp_radius'], params['lbp_points'])
    except Exception as e:
        print(f"Error applying LBP: {e}")
        features['LBP'] = image
    
    try:
        features['Textons'] = texton_features(image, params['texton_clusters'], params['texton_patch_size'])
    except Exception as e:
        print(f"Error applying Textons: {e}")
        features['Textons'] = image
    
    try:
        features['Autocorrelation'] = autocorrelation_features(image, params['autocorr_displacement'])
    except Exception as e:
        print(f"Error applying Autocorrelation: {e}")
        features['Autocorrelation'] = image
    
    try:
        features['PCA-Texture'] = pca_texture_features(image, params['pca_patch_size'], params['pca_components'])
    except Exception as e:
        print(f"Error applying PCA-Texture: {e}")
        features['PCA-Texture'] = image
    
    try:
        features['MSMD'] = msmd_features(image, params['msmd_scales'], params['msmd_directions'])
    except Exception as e:
        print(f"Error applying MSMD: {e}")
        features['MSMD'] = image
    
    return features


def create_combined_visualization(original_image, texture_features, output_path, image_name):
    """
    Create a combined visualization of original and all texture feature images.
    
    Args:
        original_image: Original input image
        texture_features: Dictionary of texture feature images
        output_path: Path to save the combined image
        image_name: Name of the original image
    """
    # Calculate grid size (original + 6 methods = 7 total)
    n_images = len(texture_features) + 1
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(16, 4 * rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Plot original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot texture feature images
    plot_idx = 1
    for method_name, feature_img in texture_features.items():
        row = plot_idx // cols
        col = plot_idx % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(feature_img, cmap='gray')
        ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_idx += 1
    
    plt.suptitle(f'Texture Analysis Results - {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the combined image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_image(args):
    """
    Process a single image with texture methods. This function is designed for multiprocessing.
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
        
        # Apply all texture methods
        texture_features = apply_all_texture_methods(image, params)
        
        # Ensure output directory exists (thread-safe)
        safe_makedirs(output_folder)
        
        # Create combined visualization and save
        create_combined_visualization(image, texture_features, output_path, image_file)
        
        return (True, image_file, subfolder_name, None)
        
    except Exception as e:
        return (False, image_file, subfolder_name, str(e))


def process_images(input_folder, params, num_processes=None):
    """
    Process all images in the input folder and its subfolders using parallel processing.
    Each image is processed individually in parallel.
    
    Args:
        input_folder: Path to input folder containing subfolders with images
        params: Dictionary containing method parameters
        num_processes: Number of processes to use (None for CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} parallel processes for individual images")
    
    # Create output folder
    base_name = os.path.basename(input_folder.rstrip('/\\'))
    output_folder = f"{input_folder}_texture"
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
                output_name = f"{os.path.splitext(image_file)[0]}_texture.png"
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
    parser = argparse.ArgumentParser(description='Apply texture analysis methods to images with parallel processing')
    
    # Input/output arguments
    parser.add_argument('input_folder', help='Input folder containing subfolders with PNG images')
    
    # Parallel processing parameters
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of parallel processes to use (default: CPU count)')
    
    # GLCM parameters
    parser.add_argument('--glcm_distances', type=int, nargs='+', default=[1],
                       help='Distances for GLCM calculation (default: [1])')
    parser.add_argument('--glcm_angles', type=int, nargs='+', default=[0, 45, 90, 135],
                       help='Angles for GLCM calculation in degrees (default: [0, 45, 90, 135])')
    
    # LBP parameters
    parser.add_argument('--lbp_radius', type=int, default=3,
                       help='Radius for LBP calculation (default: 3)')
    parser.add_argument('--lbp_points', type=int, default=24,
                       help='Number of points for LBP calculation (default: 24)')
    
    # Texton parameters
    parser.add_argument('--texton_clusters', type=int, default=16,
                       help='Number of texton clusters (default: 16)')
    parser.add_argument('--texton_patch_size', type=int, default=5,
                       help='Patch size for texton analysis (default: 5)')
    
    # Autocorrelation parameters
    parser.add_argument('--autocorr_displacement', type=int, default=10,
                       help='Maximum displacement for autocorrelation (default: 10)')
    
    # PCA parameters
    parser.add_argument('--pca_patch_size', type=int, default=8,
                       help='Patch size for PCA texture analysis (default: 8)')
    parser.add_argument('--pca_components', type=int, default=3,
                       help='Number of PCA components (default: 3)')
    
    # MSMD parameters
    parser.add_argument('--msmd_scales', type=int, nargs='+', default=[1, 2, 4],
                       help='Scales for MSMD analysis (default: [1, 2, 4])')
    parser.add_argument('--msmd_directions', type=int, nargs='+', default=[0, 45, 90, 135],
                       help='Directions for MSMD analysis in degrees (default: [0, 45, 90, 135])')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return
    
    # Validate parameters
    if args.num_processes is not None and args.num_processes < 1:
        print("Error: Number of processes must be at least 1.")
        return
    
    if args.lbp_radius < 1:
        print("Error: LBP radius must be at least 1.")
        return
    
    if args.lbp_points < 4:
        print("Error: LBP points must be at least 4.")
        return
    
    if args.texton_clusters < 2:
        print("Error: Number of texton clusters must be at least 2.")
        return
    
    # Create parameter dictionary
    params = {
        'glcm_distances': args.glcm_distances,
        'glcm_angles': args.glcm_angles,
        'lbp_radius': args.lbp_radius,
        'lbp_points': args.lbp_points,
        'texton_clusters': args.texton_clusters,
        'texton_patch_size': args.texton_patch_size,
        'autocorr_displacement': args.autocorr_displacement,
        'pca_patch_size': args.pca_patch_size,
        'pca_components': args.pca_components,
        'msmd_scales': args.msmd_scales,
        'msmd_directions': args.msmd_directions
    }
    
    print("Texture Feature Extraction (Parallelized by Individual Images)")
    print("=" * 50)
    print(f"Input folder: {args.input_folder}")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Processes to use: {args.num_processes if args.num_processes else cpu_count()}")
    print("Strategy: Each image processed in parallel with thread-safe folder creation")
    print("Texture method parameters:")
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