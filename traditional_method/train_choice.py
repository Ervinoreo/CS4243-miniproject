import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import warnings
import multiprocessing as mp
from functools import partial
import pickle
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


def downsample_feature_map(feature_map, pool_size=2):
    """
    Downsample feature map using average pooling.
    
    Args:
        feature_map: Input feature map
        pool_size: Size of pooling window (default: 2x2)
    
    Returns:
        Downsampled feature map
    """
    h, w = feature_map.shape
    new_h, new_w = h // pool_size, w // pool_size
    
    if new_h == 0 or new_w == 0:
        return feature_map
    
    pooled = np.zeros((new_h, new_w), dtype=feature_map.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            pool_region = feature_map[
                i*pool_size:(i+1)*pool_size, 
                j*pool_size:(j+1)*pool_size
            ]
            pooled[i, j] = np.mean(pool_region)
    
    return pooled


def compute_feature_statistics(feature_map):
    """
    Compute statistical features from a feature map.
    
    Args:
        feature_map: Input feature map
    
    Returns:
        List of statistical features [mean, variance, energy, entropy, power]
    """
    # Flatten the feature map
    flat_map = feature_map.flatten()
    
    # Mean
    mean_val = np.mean(flat_map)
    
    # Variance
    var_val = np.var(flat_map)
    
    # Energy
    energy_val = np.sum(flat_map ** 2) / len(flat_map)
    
    # Entropy (using histogram-based approximation)
    try:
        hist, _ = np.histogram(flat_map, bins=256, range=(0, 255))
        hist = hist + 1e-8  # Add small value to avoid log(0)
        prob = hist / np.sum(hist)
        entropy_val = -np.sum(prob * np.log2(prob))
    except:
        entropy_val = 0.0
    
    # Power (RMS)
    power_val = np.sqrt(np.mean(flat_map ** 2))
    
    return [mean_val, var_val, energy_val, entropy_val, power_val]


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


def extract_single_image_features(args):
    """
    Extract multi-layer features from a single image. This function is designed for multiprocessing.
    
    Args:
        args: Tuple of (image_path, layer_configs, params)
    
    Returns:
        Tuple of (image_path, features) or (image_path, None) if error
    """
    image_path, layer_configs, params = args
    
    try:
        # Load image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            return image_path, np.zeros(100, dtype=np.float32)  # Default fallback size
        
        all_features = []
        
        # Process original image first
        normalized_original = normalize_feature_map(original_image)
        pooled_original = downsample_feature_map(normalized_original, pool_size=2)
        flattened_original = pooled_original.flatten()
        original_stats = compute_feature_statistics(original_image)
        
        # Add original image features
        all_features.extend(flattened_original)
        all_features.extend(original_stats)
        
        # Store current layer images (start with original)
        current_layer_images = [original_image]
        
        # Process each layer
        for layer_idx, layer_config in enumerate(layer_configs):
            layer_features = []
            next_layer_images = []
            
            # Apply filters to all images from previous layer
            for img_idx, input_image in enumerate(current_layer_images):
                for filter_config in layer_config:
                    filter_name = filter_config['name']
                    filter_type = filter_config['type']
                    filter_params = filter_config.get('params', {})
                    
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
                        
                        # Normalize to [0, 1]
                        normalized_filtered = normalize_feature_map(filtered_image)
                        
                        # Downsample using 2x2 average pooling
                        pooled_filtered = downsample_feature_map(normalized_filtered, pool_size=2)
                        
                        # Flatten the pooled feature map
                        flattened_filtered = pooled_filtered.flatten()
                        
                        # Compute statistical features
                        filtered_stats = compute_feature_statistics(filtered_image)
                        
                        # Add to layer features
                        layer_features.extend(flattened_filtered)
                        layer_features.extend(filtered_stats)
                        
                        # Save filtered image for next layer
                        next_layer_images.append(filtered_image)
                        
                    except Exception as e:
                        # Add zeros if filter fails
                        dummy_size = (original_image.shape[0] // (2 ** (layer_idx + 2)), 
                                     original_image.shape[1] // (2 ** (layer_idx + 2)))
                        if dummy_size[0] > 0 and dummy_size[1] > 0:
                            dummy_features = np.zeros(dummy_size[0] * dummy_size[1] + 5)
                        else:
                            dummy_features = np.zeros(10)  # Minimal fallback
                        layer_features.extend(dummy_features)
            
            # Add layer features to all features
            all_features.extend(layer_features)
            
            # Update current layer images for next iteration
            current_layer_images = next_layer_images if next_layer_images else current_layer_images
        
        return image_path, np.array(all_features, dtype=np.float32)
    
    except Exception as e:
        return image_path, np.zeros(100, dtype=np.float32)


class FeatureExtractor:
    """
    Multi-layer feature extractor with customizable filter configurations.
    """
    
    def __init__(self, layer_configs=None, custom_params=None):
        # Set up default layer configurations if none provided
        if layer_configs is None:
            self.layer_configs = self.get_default_layer_configs()
        else:
            self.layer_configs = layer_configs
        
        # Default parameters for all filter types
        self.params = {
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
        
        # Update with custom parameters if provided
        if custom_params:
            for param_type, param_dict in custom_params.items():
                if param_type in self.params:
                    self.params[param_type].update(param_dict)
    
    def get_default_layer_configs(self):
        """
        Get the default 3-layer configuration as specified.
        
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
    
    def set_layer_configs(self, layer_configs):
        """
        Set custom layer configurations.
        
        Args:
            layer_configs: List of layers, each containing list of filter configurations
        """
        self.layer_configs = layer_configs
    
    def add_layer(self, layer_config):
        """
        Add a new layer to the configuration.
        
        Args:
            layer_config: List of filter configurations for the new layer
        """
        self.layer_configs.append(layer_config)
    
    def get_available_filters(self):
        """
        Get list of available filters by category.
        
        Returns:
            Dictionary of available filters by type
        """
        return {
            'spatial': ['Sobel', 'Laplacian', 'Gaussian', 'Low-pass', 'High-pass', 'Median', 'Butterworth'],
            'freq': ['Fourier', 'Walsh-Hadamard', 'KLT', 'Gabor', 'Power Spectrum'],
            'texture': ['GLCM', 'LBP', 'Texton', 'Autocorrelation', 'PCA-Texture', 'MSMD']
        }
    
    def extract_features_parallel(self, image_paths, n_processes=None, cache_file=None):
        """
        Extract multi-layer features from multiple images in parallel.
        
        Args:
            image_paths: List of image paths
            n_processes: Number of processes to use (default: CPU count)
            cache_file: Optional cache file to save/load features
        
        Returns:
            Dictionary mapping image_path to feature vector
        """
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(image_paths))
        
        print(f"Extracting multi-layer features from {len(image_paths)} images using {n_processes} processes...")
        print(f"Layer configuration: {len(self.layer_configs)} layers")
        for i, layer in enumerate(self.layer_configs):
            filter_names = [f"{config['name']}({config['type']})" for config in layer]
            print(f"  Layer {i+1}: {', '.join(filter_names)}")
        
        # Prepare arguments for parallel processing
        args_list = [(img_path, self.layer_configs, self.params) for img_path in image_paths]
        
        # Use multiprocessing to extract features
        feature_dict = {}
        
        with mp.Pool(n_processes) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(extract_single_image_features, args_list),
                total=len(args_list),
                desc="Extracting features"
            ))
        
        # Convert results to dictionary
        for img_path, features in results:
            feature_dict[img_path] = features
        
        # Save cache if requested
        if cache_file:
            print(f"Saving features to cache file: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(feature_dict, f)
        
        return feature_dict
    
    def extract_features(self, image):
        """
        Extract multi-layer features from a single image.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Combined feature vector from all layers
        """
        try:
            all_features = []
            
            # Process original image first
            normalized_original = normalize_feature_map(image)
            pooled_original = downsample_feature_map(normalized_original, pool_size=2)
            flattened_original = pooled_original.flatten()
            original_stats = compute_feature_statistics(image)
            
            # Add original image features
            all_features.extend(flattened_original)
            all_features.extend(original_stats)
            
            # Store current layer images (start with original)
            current_layer_images = [image]
            
            # Process each layer
            for layer_idx, layer_config in enumerate(self.layer_configs):
                layer_features = []
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
                                filtered_image = filter_func(input_image, self.params['spatial'])
                            elif filter_type == 'freq':
                                filtered_image = filter_func(input_image, self.params['freq'])
                            elif filter_type == 'texture':
                                filtered_image = filter_func(input_image, self.params['texture'])
                            else:
                                continue
                            
                            # Normalize to [0, 1]
                            normalized_filtered = normalize_feature_map(filtered_image)
                            
                            # Downsample using 2x2 average pooling
                            pooled_filtered = downsample_feature_map(normalized_filtered, pool_size=2)
                            
                            # Flatten the pooled feature map
                            flattened_filtered = pooled_filtered.flatten()
                            
                            # Compute statistical features
                            filtered_stats = compute_feature_statistics(filtered_image)
                            
                            # Add to layer features
                            layer_features.extend(flattened_filtered)
                            layer_features.extend(filtered_stats)
                            
                            # Save filtered image for next layer
                            next_layer_images.append(filtered_image)
                            
                        except Exception as e:
                            print(f"Error applying filter {filter_name}: {e}")
                            # Add zeros if filter fails
                            dummy_size = max(1, image.shape[0] // (2 ** (layer_idx + 2)))
                            dummy_features = np.zeros(dummy_size * dummy_size + 5)
                            layer_features.extend(dummy_features)
                
                # Add layer features to all features
                all_features.extend(layer_features)
                
                # Update current layer images for next iteration
                current_layer_images = next_layer_images if next_layer_images else current_layer_images
            
            return np.array(all_features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return np.zeros(100, dtype=np.float32)  # Fallback


class CharacterDataset(Dataset):
    """
    Dataset class for character images with feature extraction.
    """
    
    def __init__(self, image_paths, labels, feature_extractor, transform_features=True, 
                 n_processes=None, cache_file=None):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.transform_features = transform_features
        
        # Create label mapping
        self.label_to_idx = self._create_label_mapping()
        
        # Pre-extract features if specified
        if self.transform_features:
            if cache_file is None:
                # Generate cache filename based on dataset size and parameters
                cache_file = f"features_cache_{len(image_paths)}_images.pkl"
            
            feature_dict = self.feature_extractor.extract_features_parallel(
                self.image_paths, n_processes=n_processes, cache_file=cache_file
            )
            
            # Convert to array in the same order as image_paths
            self.features = np.array([feature_dict[img_path] for img_path in self.image_paths])
            print(f"Features extracted: {self.features.shape}")
    
    def _create_label_mapping(self):
        """Create mapping from label strings to indices."""
        unique_labels = sorted(list(set(self.labels)))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.transform_features:
            features = self.features[idx]
        else:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if image is None:
                features = np.zeros(100, dtype=np.float32)  # Default fallback size
            else:
                features = self.feature_extractor.extract_features(image)
        
        label_idx = self.label_to_idx[self.labels[idx]]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


class CharacterMLP(nn.Module):
    """
    Multi-layer Perceptron for character classification.
    """
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=36, dropout_rate=0.3):
        super(CharacterMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data(input_folder):
    """
    Load image paths and labels from folder structure.
    
    Args:
        input_folder: Path to folder containing subfolders with images
    
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    print(f"Loading data from: {input_folder}")
    
    for subfolder_name in sorted(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            print(f"Processing label: {subfolder_name}")
            
            image_files = [f for f in os.listdir(subfolder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)
                image_paths.append(image_path)
                labels.append(subfolder_name)
            
            print(f"  Found {len(image_files)} images")
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Unique labels: {sorted(list(set(labels)))}")
    
    return image_paths, labels


def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.001):
    """
    Train the MLP model.
    
    Args:
        model: The MLP model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        Training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 100

    # Determine feature suffix for model filename
    from datetime import datetime
    feature_suffix = ""
    if hasattr(model, 'layer_configs'):
        feature_suffix = f"_multilayer_{len(model.layer_configs)}layers"
    else:
        feature_suffix = "_multilayer"
    date_str = datetime.now().strftime("_%Y%m%d")

    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (features, labels) in enumerate(train_pbar):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping and best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f'best_model{feature_suffix}{date_str}.pth'
            torch.save(model.state_dict(), model_filename)
            patience_counter = 0
            print(f'  New best validation accuracy: {best_val_acc:.2f}% - Model saved as {model_filename}!')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        print('-' * 60)
    
    return history


def evaluate_model(model, test_loader, device, label_names):
    """
    Evaluate the trained model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        label_names: List of label names for confusion matrix
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy


def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


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
            filter_name, filter_type = filter_spec.strip().split(':')
            layer_config.append({
                'name': filter_name.strip(),
                'type': filter_type.strip()
            })
    return layer_config


def main():
    parser = argparse.ArgumentParser(description='Train character classifier using multi-layer features')
    parser.add_argument('input_folder', nargs='?', help='Input folder containing subfolders with character images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer sizes (default: [512, 256, 128])')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    parser.add_argument('--n_processes', type=int, default=None, 
                       help='Number of processes for feature extraction (default: CPU count)')
    parser.add_argument('--cache_dir', type=str, default='layered_feature_cache',
                       help='Directory to store feature cache files (default: layered_feature_cache)')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable feature caching')
    
    # Multi-layer configuration options
    parser.add_argument('--use_default_layers', action='store_true', default=True,
                       help='Use default 3-layer configuration (default: True)')
    parser.add_argument('--custom_layers', type=str, nargs='+',
                       help='Custom layer configurations. Format: "filter1:type1,filter2:type2" for each layer')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of feature extraction layers (default: 3)')
    
    # Available filter listing
    parser.add_argument('--list_filters', action='store_true',
                       help='List all available filters and exit')
    
    args = parser.parse_args()
    
    # List available filters if requested
    if args.list_filters:
        extractor = FeatureExtractor()
        available_filters = extractor.get_available_filters()
        print("Available filters by category:")
        for filter_type, filters in available_filters.items():
            print(f"\n{filter_type.upper()}:")
            for filter_name in filters:
                print(f"  - {filter_name}")
        print("\nExample custom layer: --custom_layers \"Sobel:spatial,Gaussian:spatial\" \"LBP:texture,Gabor:freq\"")
        return
    
    # Check if input_folder is provided for training
    if args.input_folder is None:
        parser.error("input_folder is required for training. Use --list_filters to see available filters.")
    
    # Configure layer setup
    layer_configs = None
    if args.custom_layers:
        print("Using custom layer configuration...")
        layer_configs = []
        for i, layer_str in enumerate(args.custom_layers):
            layer_config = parse_layer_config(layer_str)
            layer_configs.append(layer_config)
            print(f"  Layer {i+1}: {[f"{config['name']}({config['type']})" for config in layer_config]}")
        args.use_default_layers = False
    
    if args.use_default_layers and layer_configs is None:
        print("Using default 3-layer configuration:")
        print("  Layer 1: Sobel(spatial), Laplacian(spatial), Gaussian(spatial)")
        print("  Layer 2: LBP(texture), Autocorrelation(texture), Gabor(freq), Walsh-Hadamard(freq)")
        print("  Layer 3: MSMD(texture), High-pass(spatial), Low-pass(spatial), Butterworth(spatial)")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Print multiprocessing info
    available_cpus = mp.cpu_count()
    n_processes = args.n_processes if args.n_processes is not None else available_cpus
    print(f"Available CPUs: {available_cpus}")
    print(f"Using {n_processes} processes for feature extraction")
    
    # Load data
    image_paths, labels = load_data(args.input_folder)
    
    # Split data into train and validation sets only
    print("\nSplitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=args.val_size,
        random_state=args.random_state, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(layer_configs=layer_configs)
    
    # Setup cache files
    if not args.no_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        # Extract dataset name from input folder path
        dataset_name = os.path.basename(os.path.normpath(args.input_folder))
        # Create feature type suffix based on layer configuration
        feature_suffix = f"_multilayer_{len(feature_extractor.layer_configs)}layers"

        train_cache = os.path.join(
            args.cache_dir,
            f"{dataset_name}_train_{len(X_train)}{feature_suffix}.pkl"
        )
        val_cache = os.path.join(
            args.cache_dir,
            f"{dataset_name}_val_{len(X_val)}{feature_suffix}.pkl"
        )
    else:
        train_cache = None
        val_cache = None
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CharacterDataset(
        X_train, y_train, feature_extractor, 
        n_processes=n_processes, cache_file=train_cache
    )
    val_dataset = CharacterDataset(
        X_val, y_val, feature_extractor, 
        n_processes=n_processes, cache_file=val_cache
    )
    
    # Get feature size from first sample
    sample_features, _ = train_dataset[0]
    input_size = sample_features.shape[0]
    print(f"Feature vector size: {input_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create unique output directory for this run
    from datetime import datetime
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_suffix = f"_multilayer_{len(feature_extractor.layer_configs)}layers"
    output_dir = os.path.join("training_runs", f"run_{run_time}{feature_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    num_classes = len(train_dataset.label_to_idx)
    model = CharacterMLP(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate
    ).to(device)
    # Attach layer configuration for model filename
    model.layer_configs = feature_extractor.layer_configs

    print(f"\nModel created with {num_classes} classes")
    print(f"Model architecture: {input_size} -> {' -> '.join(map(str, args.hidden_sizes))} -> {num_classes}")

    # Train model
    def train_model_with_dir(model, train_loader, val_loader, device, num_epochs, learning_rate, output_dir, feature_suffix, run_time):
        # Patch train_model to save best model in output_dir
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 100
        date_str = datetime.now().strftime("_%Y%m%d")
        print(f"Starting training on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch_idx, (features, labels) in enumerate(train_pbar):
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for features, labels in val_pbar:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            scheduler.step(val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}]:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_filename = os.path.join(output_dir, f'best_model{feature_suffix}{date_str}.pth')
                torch.save(model.state_dict(), model_filename)
                patience_counter = 0
                print(f'  New best validation accuracy: {best_val_acc:.2f}% - Model saved as {model_filename}!')
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            print('-' * 60)
        # Save training history in output_dir
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        return history

    history = train_model_with_dir(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, learning_rate=args.learning_rate,
        output_dir=output_dir, feature_suffix=feature_suffix, run_time=run_time
    )

    # Load best model with correct feature suffix and date from output_dir
    date_str = datetime.now().strftime("_%Y%m%d")
    model_filename = os.path.join(output_dir, f'best_model{feature_suffix}{date_str}.pth')
    model.load_state_dict(torch.load(model_filename))
    
    # Get label names for future use
    label_names = sorted(list(train_dataset.label_to_idx.keys()))
    
    # Plot training history
    plot_training_history(history)

    # Save training configuration and results in output_dir
    config = {
        'input_folder': args.input_folder,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout_rate,
        'val_size': args.val_size,
        'random_state': args.random_state,
        'n_processes': n_processes,
        'input_size': input_size,
        'num_classes': num_classes,
        'label_names': label_names,
        'device': str(device),
        'layer_configs': feature_extractor.layer_configs,
        'num_layers': len(feature_extractor.layer_configs),
        'use_default_layers': args.use_default_layers,
        'custom_layers': args.custom_layers if hasattr(args, 'custom_layers') else None,
        'timestamp': datetime.now().isoformat()
    }
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best model saved as '{model_filename}'")
    print(f"Configuration saved as '{config_path}'")
    if not args.no_cache:
        print(f"Feature cache saved in '{args.cache_dir}' directory")
    print(f"All outputs for this run are in '{output_dir}'")
    print(f"Use a separate script to evaluate the model on your test set.")


if __name__ == "__main__":
    main()