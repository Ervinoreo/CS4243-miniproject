import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift, dct, idct
from scipy.linalg import hadamard
from skimage.transform import resize
from tqdm import tqdm
import threading

# Global lock for thread-safe directory creation
dir_creation_lock = threading.Lock()

def safe_makedirs(directory):
	with dir_creation_lock:
		os.makedirs(directory, exist_ok=True)

# Fourier Transform
def fourier_transform(image):
	f = np.fft.fft2(image)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
	return magnitude_spectrum.astype(np.uint8)

# Walsh-Hadamard Transform
def walsh_hadamard_transform(image):
	# Resize to square and power of 2 for Hadamard
	n = max(image.shape)
	n_pow2 = 2 ** int(np.ceil(np.log2(n)))
	img_resized = resize(image, (n_pow2, n_pow2), preserve_range=True, anti_aliasing=True).astype(np.float32)
	H = hadamard(n_pow2)
	wh = H @ img_resized @ H
	wh_norm = np.abs(wh)
	wh_norm = 255 * wh_norm / wh_norm.max()
	return wh_norm.astype(np.uint8)

# Karhunen-Loeve Transform (PCA)
def klt_transform(image, n_components=10):
    # Flatten image to 1D vector
    h, w = image.shape
    X = image.reshape(-1, 1).astype(np.float64)
    
    # Check if we have enough data
    if X.size == 0:
        return image
    
    # Center the data
    X_mean = np.mean(X)
    X_centered = X - X_mean
    
    # For single column, create a simple transformation
    if X_centered.shape[1] == 1:
        # Create patches for PCA analysis
        patch_size = min(8, min(h, w))  # Use 8x8 patches or smaller if image is small
        patches = []
        
        for i in range(0, h - patch_size + 1, patch_size//2):
            for j in range(0, w - patch_size + 1, patch_size//2):
                patch = image[i:i+patch_size, j:j+patch_size].flatten()
                patches.append(patch)
        
        if len(patches) == 0:
            return image
            
        patches = np.array(patches).T  # Each column is a patch
        
        # Center patches
        patches_mean = np.mean(patches, axis=1, keepdims=True)
        patches_centered = patches - patches_mean
        
        # Compute covariance matrix
        cov = np.cov(patches_centered)
        
        # Handle case where cov is scalar
        if cov.ndim == 0:
            cov = np.array([[cov]])
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top components
        n_components = min(n_components, eigvecs.shape[1])
        top_eigvecs = eigvecs[:, :n_components]
        
        # Project first patch onto principal components
        if patches_centered.shape[1] > 0:
            projected = top_eigvecs.T @ patches_centered[:, 0:1]
            
            # Reconstruct and reshape back to image
            reconstructed = (top_eigvecs @ projected).flatten()
            
            # Pad or crop to original image size
            if len(reconstructed) < h * w:
                reconstructed = np.pad(reconstructed, (0, h * w - len(reconstructed)))
            else:
                reconstructed = reconstructed[:h * w]
                
            result = reconstructed.reshape(h, w)
        else:
            result = image
    else:
        # Standard PCA for multi-dimensional data
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx[:n_components]]
        klt = X_centered @ eigvecs
        result = klt.reshape(h, w)
    
    # Normalize to 0-255 range
    result = np.abs(result)
    if result.max() > 0:
        result = 255 * result / result.max()
    
    return result.astype(np.uint8)

# Gabor Filters
def gabor_filter(image, frequency=0.6, theta=0):
    try:
        from skimage.filters import gabor
        filt_real, filt_imag = gabor(image, frequency=frequency, theta=theta)
        gabor_mag = np.sqrt(filt_real**2 + filt_imag**2)
        gabor_mag = 255 * gabor_mag / gabor_mag.max()
        return gabor_mag.astype(np.uint8)
    except ImportError:
        # Fallback: create simple oriented filter
        kernel_size = 15
        sigma = 3
        
        # Create coordinate grids
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Create Gabor kernel
        gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * frequency * X_rot)
        gabor_kernel = gaussian * sinusoid
        
        # Apply convolution
        from scipy.signal import convolve2d
        filtered = convolve2d(image, gabor_kernel, mode='same', boundary='symm')
        filtered = np.abs(filtered)
        filtered = 255 * filtered / (filtered.max() + 1e-8)
        return filtered.astype(np.uint8)

# Power Spectrum
def power_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2
    power = 255 * power / power.max()
    return power.astype(np.uint8)

def apply_all_transforms(image, params):
	transforms = {}
	try:
		transforms['Fourier'] = fourier_transform(image)
	except Exception as e:
		print(f"Error Fourier: {e}")
		transforms['Fourier'] = image
	try:
		transforms['Walsh-Hadamard'] = walsh_hadamard_transform(image)
	except Exception as e:
		print(f"Error Walsh-Hadamard: {e}")
		transforms['Walsh-Hadamard'] = image
	try:
		transforms['KLT'] = klt_transform(image, params['klt_components'])
	except Exception as e:
		print(f"Error KLT: {e}")
		transforms['KLT'] = image
	try:
		transforms['Gabor'] = gabor_filter(image, params['gabor_frequency'], params['gabor_theta'])
	except Exception as e:
		print(f"Error Gabor: {e}")
		transforms['Gabor'] = image
	try:
		transforms['Power Spectrum'] = power_spectrum(image)
	except Exception as e:
		print(f"Error Power Spectrum: {e}")
		transforms['Power Spectrum'] = image
	return transforms

def create_combined_visualization(original_image, transformed_images, output_path, image_name):
	n_images = len(transformed_images) + 1
	cols = 3
	rows = (n_images + cols - 1) // cols
	fig = plt.figure(figsize=(16, 3 * rows))
	gs = GridSpec(rows, cols, figure=fig)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.imshow(original_image, cmap='gray')
	ax1.set_title('Original', fontsize=12, fontweight='bold')
	ax1.axis('off')
	plot_idx = 1
	for name, img in transformed_images.items():
		row = plot_idx // cols
		col = plot_idx % cols
		ax = fig.add_subplot(gs[row, col])
		ax.imshow(img, cmap='gray')
		ax.set_title(f'{name}', fontsize=12, fontweight='bold')
		ax.axis('off')
		plot_idx += 1
	plt.suptitle(f'Frequency-Domain Results - {image_name}', fontsize=16, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches='tight')
	plt.close()

def process_single_image(args):
	image_path, output_path, output_folder, params = args
	image_file = os.path.basename(image_path)
	subfolder_name = os.path.basename(os.path.dirname(image_path))
	try:
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		if image is None:
			return (False, image_file, subfolder_name, "Could not read image")
		transformed_images = apply_all_transforms(image, params)
		safe_makedirs(output_folder)
		create_combined_visualization(image, transformed_images, output_path, image_file)
		return (True, image_file, subfolder_name, None)
	except Exception as e:
		return (False, image_file, subfolder_name, str(e))

def process_images(input_folder, params, num_processes=None):
	if num_processes is None:
		from multiprocessing import cpu_count
		num_processes = cpu_count()
	print(f"Using {num_processes} parallel processes for individual images")
	base_name = os.path.basename(input_folder.rstrip('/\\'))
	output_folder = f"{input_folder}_freq_dom"
	os.makedirs(output_folder, exist_ok=True)
	print(f"Processing images from: {input_folder}")
	print(f"Output folder: {output_folder}")
	image_tasks = []
	for subfolder_name in os.listdir(input_folder):
		subfolder_path = os.path.join(input_folder, subfolder_name)
		if os.path.isdir(subfolder_path):
			output_subfolder = os.path.join(output_folder, subfolder_name)
			image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.png')]
			for image_file in image_files:
				image_path = os.path.join(subfolder_path, image_file)
				output_name = f"{os.path.splitext(image_file)[0]}_freqdom.png"
				output_path = os.path.join(output_subfolder, output_name)
				image_tasks.append((image_path, output_path, output_subfolder, params))
	if not image_tasks:
		print("No images found to process.")
		return
	from multiprocessing import Pool
	import time
	start_time = time.time()
	total_processed = 0
	total_images = len(image_tasks)
	subfolder_stats = {}
	print(f"Processing {total_images} images in parallel...")
	with Pool(processes=num_processes) as pool:
		with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
			for result in pool.imap(process_single_image, image_tasks):
				success, image_file, subfolder_name, error_msg = result
				if subfolder_name not in subfolder_stats:
					subfolder_stats[subfolder_name] = {'processed': 0, 'total': 0, 'errors': []}
				subfolder_stats[subfolder_name]['total'] += 1
				if success:
					total_processed += 1
					subfolder_stats[subfolder_name]['processed'] += 1
				else:
					subfolder_stats[subfolder_name]['errors'].append(f"{image_file}: {error_msg}")
					print(f"    Error processing {image_file} in {subfolder_name}: {error_msg}")
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
	print(f"\nSubfolder Statistics:")
	for subfolder_name, stats in subfolder_stats.items():
		print(f"  {subfolder_name}: {stats['processed']}/{stats['total']} images processed")
		if stats['errors']:
			print(f"    Errors: {len(stats['errors'])}")
	print(f"Results saved to: {output_folder}")
	print("="*50)

def main():
	parser = argparse.ArgumentParser(description='Apply frequency-domain transforms to images with parallel processing')
	parser.add_argument('input_folder', help='Input folder containing subfolders with PNG images')
	parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes to use (default: CPU count)')
	parser.add_argument('--klt_components', type=int, default=10, help='Number of KLT components (default: 10)')
	parser.add_argument('--gabor_frequency', type=float, default=0.6, help='Gabor filter frequency (default: 0.6)')
	parser.add_argument('--gabor_theta', type=float, default=0.0, help='Gabor filter orientation theta in radians (default: 0.0)')
	args = parser.parse_args()
	if not os.path.isdir(args.input_folder):
		print(f"Error: Input folder '{args.input_folder}' does not exist.")
		return
	if args.klt_components < 1:
		print("Error: KLT components must be >= 1.")
		return

	params = {
		'klt_components': args.klt_components,
		'gabor_frequency': args.gabor_frequency,
		'gabor_theta': args.gabor_theta
	}
	print("Frequency-Domain Feature Extraction (Parallelized by Individual Images)")
	print("=" * 50)
	print(f"Input folder: {args.input_folder}")
	from multiprocessing import cpu_count
	print(f"Available CPU cores: {cpu_count()}")
	print(f"Processes to use: {args.num_processes if args.num_processes else cpu_count()}")
	print("Strategy: Each image processed in parallel with thread-safe folder creation")
	print("Transform parameters:")
	for param, value in params.items():
		print(f"  {param}: {value}")
	print("=" * 50)
	process_images(args.input_folder, params, args.num_processes)

if __name__ == "__main__":
	from multiprocessing import freeze_support
	freeze_support()
	main()
