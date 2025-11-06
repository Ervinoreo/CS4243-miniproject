import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Import feature extraction functions
from local_spatial_feature import apply_all_filters
from freq_domain_feature import apply_all_transforms
from texture_feature import apply_all_texture_methods

# Import individual filter functions
from local_spatial_feature import (
    low_pass_filter, gaussian_filter, high_pass_filter, laplacian_filter, 
    sobel_filter, median_filter, butterworth_filter
)
from freq_domain_feature import (
    fourier_transform, walsh_hadamard_transform, klt_transform, 
    gabor_filter, power_spectrum
)
from texture_feature import (
    glcm_features, lbp_features, texton_features, autocorrelation_features, 
    pca_texture_features, msmd_features
)


def normalize_feature_map(feature_map):
    """
    Normalize feature map to [0, 1] range.
    
    Args:
        feature_map: Input feature map
    
    Returns:
        Normalized feature map in [0, 1] range
    """
    fmin, fmax = feature_map.min(), feature_map.max()
    if fmax > fmin:
        return (feature_map - fmin) / (fmax - fmin)
    return feature_map


def get_filter_function(filter_name, filter_type):
    """
    Get the appropriate filter function based on name and type.
    
    Args:
        filter_name: Name of the filter
        filter_type: Type ('spatial', 'freq', 'texture')
    
    Returns:
        Filter function
    """
    filter_mapping = {
        'spatial': {
            'Sobel': lambda img, params: sobel_filter(img, params.get('sobel_dx', 1), params.get('sobel_dy', 1), params.get('sobel_kernel', 3)),
            'Laplacian': lambda img, params: laplacian_filter(img, params.get('laplacian_kernel', 3)),
            'Gaussian': lambda img, params: gaussian_filter(img, params.get('gaussian_kernel', 5), params.get('gaussian_sigma', 1.0)),
            'Low-pass': lambda img, params: low_pass_filter(img, params.get('low_pass_cutoff', 30.0)),
            'High-pass': lambda img, params: high_pass_filter(img, params.get('high_pass_cutoff', 30.0)),
            'Median': lambda img, params: median_filter(img, params.get('median_kernel', 5)),
            'Butterworth': lambda img, params: butterworth_filter(img, params.get('butter_cutoff', 30.0), params.get('butter_order', 2), params.get('butter_type', 'low'))
        },
        'freq': {
            'Fourier': lambda img, params: fourier_transform(img),
            'Walsh-Hadamard': lambda img, params: walsh_hadamard_transform(img),
            'KLT': lambda img, params: klt_transform(img, params.get('klt_components', 10)),
            'Gabor': lambda img, params: gabor_filter(img, params.get('gabor_frequency', 0.6), params.get('gabor_theta', 0.0)),
            'Power Spectrum': lambda img, params: power_spectrum(img)
        },
        'texture': {
            'GLCM': lambda img, params: glcm_features(img, params.get('glcm_distances', [1]), params.get('glcm_angles', [0, 45, 90, 135])),
            'LBP': lambda img, params: lbp_features(img, params.get('lbp_radius', 3), params.get('lbp_points', 24)),
            'Texton': lambda img, params: texton_features(img, params.get('texton_clusters', 16), params.get('texton_patch_size', 5)),
            'Autocorrelation': lambda img, params: autocorrelation_features(img, params.get('autocorr_displacement', 10)),
            'PCA-Texture': lambda img, params: pca_texture_features(img, params.get('pca_patch_size', 8), params.get('pca_components', 3)),
            'MSMD': lambda img, params: msmd_features(img, params.get('msmd_scales', [1, 2, 4]), params.get('msmd_directions', [0, 45, 90, 135]))
        }
    }
    
    return filter_mapping.get(filter_type, {}).get(filter_name)


def get_default_layer_configs():
    """
    Get the default 3-layer configuration.
    
    Layer 1: Sobel, Laplacian, Gaussian (applied to original image)
    Layer 2: LBP, Autocorrelation, Gabor, Walsh-Hadamard (applied to Layer 1 features)
    Layer 3: MSMD, High-pass, Low-pass, Butterworth (applied to Layer 2 features)
    """
    return [
        [  # Layer 1
            {'name': 'Sobel', 'type': 'spatial'},
            {'name': 'Laplacian', 'type': 'spatial'},
            {'name': 'Gaussian', 'type': 'spatial'}
        ],
        [  # Layer 2
            {'name': 'LBP', 'type': 'texture'},
            {'name': 'Autocorrelation', 'type': 'texture'},
            {'name': 'Gabor', 'type': 'freq'},
            {'name': 'Walsh-Hadamard', 'type': 'freq'}
        ],
        [  # Layer 3
            {'name': 'MSMD', 'type': 'texture'},
            {'name': 'High-pass', 'type': 'spatial'},
            {'name': 'Low-pass', 'type': 'spatial'},
            {'name': 'Butterworth', 'type': 'spatial'}
        ]
    ]


def get_default_params():
    """
    Get default parameters for all filter types.
    """
    return {
        'spatial': {
            'low_pass_cutoff': 30.0,
            'gaussian_kernel': 5,
            'gaussian_sigma': 1.0,
            'high_pass_cutoff': 30.0,
            'laplacian_kernel': 3,
            'sobel_dx': 1,
            'sobel_dy': 1,
            'sobel_kernel': 3,
            'median_kernel': 5,
            'butter_cutoff': 30.0,
            'butter_order': 2,
            'butter_type': 'low'
        },
        'freq': {
            'klt_components': 10,
            'gabor_frequency': 0.6,
            'gabor_theta': 0.0
        },
        'texture': {
            'glcm_distances': [1],
            'glcm_angles': [0, 45, 90, 135],
            'lbp_radius': 3,
            'lbp_points': 24,
            'texton_clusters': 16,
            'texton_patch_size': 5,
            'autocorr_displacement': 10,
            'pca_patch_size': 8,
            'pca_components': 3,
            'msmd_scales': [1, 2, 4],
            'msmd_directions': [0, 45, 90, 135]
        }
    }


def create_layer_visualization(original_image, filtered_images, layer_idx, output_path, image_name):
    """
    Create a visualization for a specific layer showing original and all filtered images.
    
    Args:
        original_image: Original input image (either the source or previous layer result)
        filtered_images: Dictionary of {filter_name: filtered_image}
        layer_idx: Layer index (1, 2, 3, etc.)
        output_path: Path to save the combined image
        image_name: Name of the original image
    """
    # Calculate grid size (original + filtered images)
    n_images = len(filtered_images) + 1
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Plot original/input image
    ax1 = fig.add_subplot(gs[0, 0])
    if len(original_image.shape) == 2:  # Grayscale
        ax1.imshow(original_image, cmap='gray')
    else:  # Color or multi-channel - show as grayscale
        if original_image.shape[2] > 1:
            display_img = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2GRAY) if original_image.shape[2] == 3 else original_image[:,:,0]
        else:
            display_img = original_image[:,:,0]
        ax1.imshow(display_img, cmap='gray')
    ax1.set_title(f'Input to Layer {layer_idx}', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # Plot filtered images
    plot_idx = 1
    for filter_name, filtered_img in filtered_images.items():
        row = plot_idx // cols
        col = plot_idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Handle different types of filtered results
        if isinstance(filtered_img, np.ndarray):
            if len(filtered_img.shape) == 2:  # 2D array
                ax.imshow(filtered_img, cmap='gray')
            elif len(filtered_img.shape) == 1:  # 1D feature vector
                # Reshape to square if possible, otherwise show as 1D plot
                size = int(np.sqrt(len(filtered_img)))
                if size * size == len(filtered_img):
                    ax.imshow(filtered_img.reshape(size, size), cmap='gray')
                else:
                    ax.plot(filtered_img)
                    ax.set_title(f'Layer {layer_idx}: {filter_name}\n(Feature Vector)', fontsize=9, fontweight='bold')
                    plot_idx += 1
                    continue
            else:  # Multi-channel - show first channel
                ax.imshow(filtered_img[:,:,0] if filtered_img.shape[2] > 0 else filtered_img, cmap='gray')
        else:
            # If it's not an array, create a placeholder
            ax.text(0.5, 0.5, f'Layer {layer_idx}: {filter_name}\n(Non-image result)', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f'Layer {layer_idx}: {filter_name}', fontsize=9, fontweight='bold')
        ax.axis('off')
        plot_idx += 1
    
    plt.suptitle(f'Layer {layer_idx} Feature Extraction - {image_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save the combined image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_image_visualization(args):
    """
    Process a single image and create visualizations for each layer.
    
    Args:
        args: Tuple of (image_path, output_base_folder, layer_configs, params)
    
    Returns:
        Tuple of (success, image_name, subfolder_name, error_message)
    """
    image_path, output_base_folder, layer_configs, params = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    subfolder_name = os.path.basename(os.path.dirname(image_path))
    
    try:
        # Load original image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            return (False, image_name, subfolder_name, "Could not read image")
        
        # Create output directory structure
        image_output_folder = os.path.join(output_base_folder, subfolder_name, image_name)
        os.makedirs(image_output_folder, exist_ok=True)
        
        # Save original image
        original_output_path = os.path.join(image_output_folder, f"ori_{image_name}.png")
        cv2.imwrite(original_output_path, original_image)
        
        # Store current layer images (start with original)
        current_layer_images = [original_image]
        
        # Process each layer
        for layer_idx, layer_config in enumerate(layer_configs, 1):
            layer_filtered_images = {}
            next_layer_images = []
            
            # Apply filters to all images from previous layer
            for img_idx, input_image in enumerate(current_layer_images):
                for filter_config in layer_config:
                    filter_name = filter_config['name']
                    filter_type = filter_config['type']
                    
                    try:
                        # Get the appropriate filter function
                        filter_func = get_filter_function(filter_name, filter_type)
                        if filter_func is None:
                            continue
                        
                        # Apply the filter
                        if filter_type == 'spatial':
                            filtered_image = filter_func(input_image, params['spatial'])
                        elif filter_type == 'freq':
                            filtered_image = filter_func(input_image, params['freq'])
                        elif filter_type == 'texture':
                            filtered_image = filter_func(input_image, params['texture'])
                        else:
                            continue
                        
                        # Handle different result types
                        if isinstance(filtered_image, np.ndarray):
                            if len(filtered_image.shape) >= 2:
                                # If it's a 2D+ array, normalize for visualization
                                if filtered_image.dtype != np.uint8:
                                    normalized_filtered = normalize_feature_map(filtered_image)
                                    filtered_image = (normalized_filtered * 255).astype(np.uint8)
                            
                        # Store for visualization (use input image index in key if multiple inputs)
                        key = f"{filter_name}" if len(current_layer_images) == 1 else f"{filter_name}_img{img_idx}"
                        layer_filtered_images[key] = filtered_image
                        
                        # Save filtered image for next layer
                        if isinstance(filtered_image, np.ndarray) and len(filtered_image.shape) >= 2:
                            next_layer_images.append(filtered_image)
                        
                    except Exception as e:
                        print(f"  Warning: Filter {filter_name} failed on layer {layer_idx}: {str(e)}")
                        continue
            
            # Create layer visualization if we have any filtered images
            if layer_filtered_images:
                # Use the first input image as the "original" for this layer
                layer_input_image = current_layer_images[0] if current_layer_images else original_image
                layer_output_path = os.path.join(image_output_folder, f"{image_name}_layer{layer_idx}.png")
                create_layer_visualization(layer_input_image, layer_filtered_images, layer_idx, layer_output_path, image_name)
            
            # Update current layer images for next iteration
            if next_layer_images:
                current_layer_images = next_layer_images
            else:
                # If no images for next layer, stop processing
                break
        
        return (True, image_name, subfolder_name, None)
        
    except Exception as e:
        return (False, image_name, subfolder_name, str(e))


def parse_layer_config(layer_str):
    """
    Parse layer configuration string.
    
    Format: "filter1:type1,filter2:type2,..."
    Example: "Sobel:spatial,Laplacian:spatial,Gaussian:spatial"
    """
    if not layer_str:
        return []
    
    layer_config = []
    for filter_spec in layer_str.split(','):
        if ':' in filter_spec:
            filter_name, filter_type = filter_spec.split(':', 1)
            layer_config.append({'name': filter_name.strip(), 'type': filter_type.strip()})
    return layer_config


def process_all_images(input_folder, output_folder, layer_configs, params, n_processes=None):
    """
    Process all images in the input folder using parallel processing.
    
    Args:
        input_folder: Input folder containing subfolders with images
        output_folder: Output folder for visualizations
        layer_configs: List of layer configurations
        params: Parameters for filters
        n_processes: Number of processes to use
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"Using {n_processes} parallel processes")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing images from: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Collect all image files
    image_tasks = []
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Get all image files in the subfolder
            image_files = [f for f in os.listdir(subfolder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)
                # Add to task list: (image_path, output_folder, layer_configs, params)
                image_tasks.append((image_path, output_folder, layer_configs, params))
    
    if not image_tasks:
        print("No images found to process.")
        return
    
    total_images = len(image_tasks)
    print(f"Found {total_images} images to process")
    
    # Process all images in parallel
    success_count = 0
    with mp.Pool(processes=n_processes) as pool:
        with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
            for result in pool.imap(process_single_image_visualization, image_tasks):
                success, image_name, subfolder_name, error_msg = result
                
                if success:
                    success_count += 1
                else:
                    print(f"  Error processing {image_name} in {subfolder_name}: {error_msg}")
                
                pbar.set_postfix({
                    'Success': f"{success_count}/{total_images}",
                    'Current': subfolder_name[:8] + '...' if len(subfolder_name) > 8 else subfolder_name
                })
                pbar.update(1)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{total_images} images")
    print(f"Results saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-layer feature extraction on images')
    
    # Input/output arguments
    parser.add_argument('input_folder', help='Input folder containing subfolders with images')
    parser.add_argument('--output_folder', type=str, default=None, 
                       help='Output folder for visualizations (default: input_folder + "_multilayer_vis")')
    
    # Layer configuration options
    parser.add_argument('--use_default_layers', action='store_true', default=True,
                       help='Use default 3-layer configuration (default: True)')
    parser.add_argument('--custom_layers', type=str, nargs='+',
                       help='Custom layer configurations. Format: "filter1:type1,filter2:type2" for each layer')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of feature extraction layers (default: 3, only used with default layers)')
    
    # Parallel processing
    parser.add_argument('--n_processes', type=int, default=None, 
                       help='Number of processes for parallel processing (default: CPU count)')
    
    # Available filter listing
    parser.add_argument('--list_filters', action='store_true',
                       help='List all available filters and exit')
    
    args = parser.parse_args()
    
    # List available filters if requested
    if args.list_filters:
        print("Available filters by category:")
        print("\nSpatial filters:")
        print("  Sobel, Laplacian, Gaussian, Low-pass, High-pass, Median, Butterworth")
        print("\nFrequency domain filters:")
        print("  Fourier, Walsh-Hadamard, KLT, Gabor, Power Spectrum")
        print("\nTexture filters:")
        print("  GLCM, LBP, Texton, Autocorrelation, PCA-Texture, MSMD")
        print("\nExample custom layer: --custom_layers \"Sobel:spatial,Gaussian:spatial\" \"LBP:texture,Gabor:freq\"")
        return
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return
    
    # Configure output folder
    if args.output_folder is None:
        base_name = os.path.basename(args.input_folder.rstrip('/\\'))
        args.output_folder = f"{args.input_folder}_multilayer_vis"
    
    # Configure layer setup
    layer_configs = None
    if args.custom_layers:
        layer_configs = []
        for layer_str in args.custom_layers:
            layer_config = parse_layer_config(layer_str)
            if layer_config:
                layer_configs.append(layer_config)
        print(f"Using {len(layer_configs)} custom layers")
    
    if args.use_default_layers and layer_configs is None:
        layer_configs = get_default_layer_configs()[:args.num_layers]
        print(f"Using default {len(layer_configs)} layers")
    
    if not layer_configs:
        print("Error: No layer configurations specified")
        return
    
    # Get default parameters
    params = get_default_params()
    
    # Print configuration
    print("Multi-layer Feature Visualization")
    print("=" * 50)
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Number of layers: {len(layer_configs)}")
    for i, layer in enumerate(layer_configs, 1):
        filters = [f"{cfg['name']}({cfg['type']})" for cfg in layer]
        print(f"  Layer {i}: {', '.join(filters)}")
    print("=" * 50)
    
    # Process images
    process_all_images(args.input_folder, args.output_folder, layer_configs, params, args.n_processes)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()
