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


def extract_single_image_features(args):
    """
    Extract features from a single image. This function is designed for multiprocessing.
    
    Args:
        args: Tuple of (image_path, feature_extractor_params)
    
    Returns:
        Tuple of (image_path, features) or (image_path, None) if error
    """
    image_path, (spatial_params, freq_params, texture_params, use_spatial, use_freq, use_texture) = args
    
    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return image_path, np.zeros(103, dtype=np.float32)
        
        features = []
        
        # 1. Local Spatial Features
        if use_spatial:
            try:
                spatial_features = apply_all_filters(image, spatial_params)
                for feature_name, feature_img in spatial_features.items():
                    features.extend([
                        np.mean(feature_img),
                        np.std(feature_img),
                        np.min(feature_img),
                        np.max(feature_img),
                        np.median(feature_img)
                    ])
            except Exception as e:
                features.extend([0] * (7 * 5))
        
        # 2. Frequency Domain Features
        if use_freq:
            try:
                freq_features = apply_all_transforms(image, freq_params)
                for feature_name, feature_img in freq_features.items():
                    features.extend([
                        np.mean(feature_img),
                        np.std(feature_img),
                        np.min(feature_img),
                        np.max(feature_img),
                        np.median(feature_img)
                    ])
            except Exception as e:
                features.extend([0] * (6 * 5))
        
        # 3. Texture Features
        if use_texture:
            try:
                texture_features = apply_all_texture_methods(image, texture_params)
                for feature_name, feature_img in texture_features.items():
                    features.extend([
                        np.mean(feature_img),
                        np.std(feature_img),
                        np.min(feature_img),
                        np.max(feature_img),
                        np.median(feature_img)
                    ])
            except Exception as e:
                features.extend([0] * (6 * 5))
        
        # 4. Raw pixel statistics
        try:
            features.extend([
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image),
                np.median(image),
                np.sum(image > 127),
                np.sum(image < 64),
                np.mean(np.diff(image.flatten())),
            ])
        except Exception as e:
            features.extend([0] * 8)
        
        return image_path, np.array(features, dtype=np.float32)
    
    except Exception as e:
        # Calculate expected feature size based on enabled features
        feature_size = 8  # Raw pixel statistics
        if use_spatial:
            feature_size += 7 * 5
        if use_freq:
            feature_size += 6 * 5
        if use_texture:
            feature_size += 6 * 5
        return image_path, np.zeros(feature_size, dtype=np.float32)


class FeatureExtractor:
    """
    Combined feature extractor using local spatial, frequency domain, and texture features.
    """
    
    def __init__(self, use_spatial=True, use_freq=True, use_texture=True):
        # Feature flags
        self.use_spatial = use_spatial
        self.use_freq = use_freq
        self.use_texture = use_texture
        
        # Default parameters for feature extraction
        self.spatial_params = {
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
        }
        
        self.freq_params = {
            'klt_components': 10,
            'wavelet_type': 'haar',
            'wavelet_level': 2,
            'gabor_frequency': 0.6,
            'gabor_theta': 0.0
        }
        
        self.texture_params = {
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
    
    def extract_features_parallel(self, image_paths, n_processes=None, cache_file=None):
        """
        Extract features from multiple images in parallel.
        
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
        
        print(f"Extracting features from {len(image_paths)} images using {n_processes} processes...")
        print(f"Feature extraction enabled: Spatial={self.use_spatial}, Frequency={self.use_freq}, Texture={self.use_texture}")
        
        # Prepare arguments for parallel processing
        params = (self.spatial_params, self.freq_params, self.texture_params, 
                 self.use_spatial, self.use_freq, self.use_texture)
        args_list = [(img_path, params) for img_path in image_paths]
        
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
        Extract all features from an image.
        
        Args:
            image: Input grayscale image (28x28)
        
        Returns:
            Combined feature vector
        """
        features = []
        
        if self.use_spatial:
            try:
                # 1. Local Spatial Features
                spatial_features = apply_all_filters(image, self.spatial_params)
                for feature_name, feature_img in spatial_features.items():
                    # Extract statistical features from each filtered image
                    features.extend([
                        np.mean(feature_img),
                        np.std(feature_img),
                        np.min(feature_img),
                        np.max(feature_img),
                        np.median(feature_img)
                    ])
            except Exception as e:
                print(f"Error extracting spatial features: {e}")
                features.extend([0] * (7 * 5))  # 7 filters * 5 statistics each
        
        if self.use_freq:
            try:
                # 2. Frequency Domain Features
                freq_features = apply_all_transforms(image, self.freq_params)
                for feature_name, feature_img in freq_features.items():
                    # Extract statistical features from each transformed image
                    features.extend([
                        np.mean(feature_img),
                        np.std(feature_img),
                        np.min(feature_img),
                        np.max(feature_img),
                        np.median(feature_img)
                    ])
            except Exception as e:
                print(f"Error extracting frequency features: {e}")
                features.extend([0] * (6 * 5))  # 6 transforms * 5 statistics each
        
        if self.use_texture:
            try:
                # 3. Texture Features
                texture_features = apply_all_texture_methods(image, self.texture_params)
                for feature_name, feature_img in texture_features.items():
                    # Extract statistical features from each texture analysis
                    features.extend([
                        np.mean(feature_img),
                        np.std(feature_img),
                        np.min(feature_img),
                        np.max(feature_img),
                        np.median(feature_img)
                    ])
            except Exception as e:
                print(f"Error extracting texture features: {e}")
                features.extend([0] * (6 * 5))  # 6 texture methods * 5 statistics each
        
        # 4. Add raw pixel statistics as baseline
        try:
            features.extend([
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image),
                np.median(image),
                np.sum(image > 127),  # Number of bright pixels
                np.sum(image < 64),   # Number of dark pixels
                np.mean(np.diff(image.flatten())),  # Average pixel difference
            ])
        except Exception as e:
            print(f"Error extracting raw pixel features: {e}")
            features.extend([0] * 8)
        
        return np.array(features, dtype=np.float32)


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
                # Calculate expected feature size based on enabled features
                feature_size = 8  # Raw pixel statistics
                if self.feature_extractor.use_spatial:
                    feature_size += 7 * 5
                if self.feature_extractor.use_freq:
                    feature_size += 6 * 5
                if self.feature_extractor.use_texture:
                    feature_size += 6 * 5
                features = np.zeros(feature_size, dtype=np.float32)
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
    if hasattr(model, 'feature_flags'):
        if model.feature_flags.get('use_spatial', False):
            feature_suffix += "_spatial"
        if model.feature_flags.get('use_freq', False):
            feature_suffix += "_freq"
        if model.feature_flags.get('use_texture', False):
            feature_suffix += "_texture"
    else:
        feature_suffix = ""
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


def main():
    parser = argparse.ArgumentParser(description='Train character classifier using combined features')
    parser.add_argument('input_folder', help='Input folder containing subfolders with character images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer sizes (default: [512, 256, 128])')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    parser.add_argument('--n_processes', type=int, default=None, 
                       help='Number of processes for feature extraction (default: CPU count)')
    parser.add_argument('--cache_dir', type=str, default='feature_cache',
                       help='Directory to store feature cache files (default: feature_cache)')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable feature caching')
    parser.add_argument('--use_spatial', action='store_true', default=False,
                       help='Use local spatial features (default: False, use --use_spatial to enable)')
    parser.add_argument('--use_freq', action='store_true', default=False,
                       help='Use frequency domain features (default: False, use --use_freq to enable)')
    parser.add_argument('--use_texture', action='store_true', default=False,
                       help='Use texture features (default: False, use --use_texture to enable)')
    
    args = parser.parse_args()
    
    # If no feature flags are set, enable all features by default
    if not (args.use_spatial or args.use_freq or args.use_texture):
        print("No feature flags specified. Enabling all features by default.")
        args.use_spatial = True
        args.use_freq = True
        args.use_texture = True
    
    print(f"Feature extraction configuration:")
    print(f"  - Spatial features: {args.use_spatial}")
    print(f"  - Frequency domain features: {args.use_freq}")
    print(f"  - Texture features: {args.use_texture}")
    
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
    feature_extractor = FeatureExtractor(
        use_spatial=args.use_spatial,
        use_freq=args.use_freq,
        use_texture=args.use_texture
    )
    
    # Setup cache files
    if not args.no_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        # Extract dataset name from input folder path
        dataset_name = os.path.basename(os.path.normpath(args.input_folder))
        # Create feature type suffix
        feature_suffix = ""
        if args.use_spatial:
            feature_suffix += "_spatial"
        if args.use_freq:
            feature_suffix += "_freq"
        if args.use_texture:
            feature_suffix += "_texture"

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
    
    # Create model
    num_classes = len(train_dataset.label_to_idx)
    model = CharacterMLP(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate
    ).to(device)
    # Attach feature flags for model filename
    model.feature_flags = {
        'use_spatial': args.use_spatial,
        'use_freq': args.use_freq,
        'use_texture': args.use_texture
    }

    print(f"\nModel created with {num_classes} classes")
    print(f"Model architecture: {input_size} -> {' -> '.join(map(str, args.hidden_sizes))} -> {num_classes}")

    # Train model
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, learning_rate=args.learning_rate
    )

    # Load best model with correct feature suffix and date
    from datetime import datetime
    feature_suffix = ""
    if args.use_spatial:
        feature_suffix += "_spatial"
    if args.use_freq:
        feature_suffix += "_freq"
    if args.use_texture:
        feature_suffix += "_texture"
    date_str = datetime.now().strftime("_%Y%m%d")
    model_filename = f'best_model{feature_suffix}{date_str}.pth'
    model.load_state_dict(torch.load(model_filename))
    
    # Get label names for future use
    label_names = sorted(list(train_dataset.label_to_idx.keys()))
    
    # Plot training history
    plot_training_history(history)
    
    # Save training configuration and results
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
        'use_spatial': args.use_spatial,
        'use_freq': args.use_freq,
        'use_texture': args.use_texture,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best model saved as '{model_filename}'")
    print(f"Configuration saved as 'training_config.json'")
    if not args.no_cache:
        print(f"Feature cache saved in '{args.cache_dir}' directory")
    print(f"Use a separate script to evaluate the model on your test set.")


if __name__ == "__main__":
    main()