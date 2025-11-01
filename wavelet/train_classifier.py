import os
import cv2
import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from tqdm import tqdm


# Wavelet feature extraction (same as wavelet_visual.py)
wavelets = ["haar", "db2", "sym2"]
max_level = 3


def extract_wavelet_features(arr):
    """Extract statistical features from wavelet coefficients"""
    arr_abs = np.abs(arr)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "energy": np.sum(np.square(arr)),
        "entropy": -np.sum(np.where(arr_abs > 0, arr_abs * np.log2(arr_abs + 1e-12), 0)),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def wavelet_features(img, wavelet, levels=max_level):
    """Extract comprehensive wavelet features from an image"""
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    features = {}
    
    for lvl, coeff in enumerate(coeffs):
        if lvl == 0:
            # LL subband (approximation)
            LL = coeff
            stats = extract_wavelet_features(LL)
            for k, v in stats.items():
                features[f"{wavelet}_L{lvl}_LL_{k}"] = v
        else:
            # Detail subbands (LH, HL, HH)
            LH, HL, HH = coeff
            subbands = {"LH": LH, "HL": HL, "HH": HH}
            for name, band in subbands.items():
                stats = extract_wavelet_features(band)
                for k, v in stats.items():
                    features[f"{wavelet}_L{lvl}_{name}_{k}"] = v
    
    return features


def extract_all_wavelet_features(img_path):
    """Extract all wavelet features from a single image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    all_features = {}
    
    # Extract features for each wavelet
    for wavelet in wavelets:
        features = wavelet_features(img, wavelet, levels=max_level)
        all_features.update(features)
    
    return all_features


class WaveletDataset(Dataset):
    """Custom dataset for wavelet features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification"""
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data(input_folder):
    """Load images and labels from folder structure"""
    print("Loading and processing images...")
    
    all_features = []
    all_labels = []
    
    # Get all class folders (0-9, a-z)
    class_folders = []
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path):
            class_folders.append(item)
    
    class_folders.sort()  # Ensure consistent ordering
    print(f"Found classes: {class_folders}")
    
    # Process each class folder
    for class_name in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(input_folder, class_name)
        
        # Process each image in the class folder
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, filename)
                
                # Extract wavelet features
                features = extract_all_wavelet_features(img_path)
                if features is not None:
                    # Convert features dict to list (maintaining order)
                    feature_vector = list(features.values())
                    all_features.append(feature_vector)
                    all_labels.append(class_name)
    
    print(f"Loaded {len(all_features)} images from {len(class_folders)} classes")
    return np.array(all_features), np.array(all_labels), class_folders


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the MLP model"""
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
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
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 50)
    
    return best_model_state, history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier on wavelet features")
    parser.add_argument("input_folder", type=str, help="Path to input folder with class subfolders")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--hidden_sizes", nargs='+', type=int, default=[512, 256, 128], 
                       help="Hidden layer sizes (e.g., --hidden_sizes 512 256 128)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split ratio (default 0.2 for 20%)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    features, labels, class_names = load_data(args.input_folder)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data (20% test, 80% train)
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, encoded_labels, 
        test_size=args.test_split, 
        random_state=42, 
        stratify=encoded_labels
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Feature dimension: {features_scaled.shape[1]}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create datasets and dataloaders
    train_dataset = WaveletDataset(X_train, y_train)
    val_dataset = WaveletDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    input_size = features_scaled.shape[1]
    num_classes = len(class_names)
    
    model = MLPClassifier(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    print(f"Model architecture: {input_size} -> {' -> '.join(map(str, args.hidden_sizes))} -> {num_classes}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    best_model_state, history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.epochs, device
    )
    
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"wavelet_classifier_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save best model
    model.load_state_dict(best_model_state)
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'input_size': input_size,
            'hidden_sizes': args.hidden_sizes,
            'num_classes': num_classes,
            'dropout_rate': args.dropout_rate
        },
        'class_names': class_names,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'best_val_acc': best_val_acc
    }, os.path.join(output_folder, "best_model.pth"))
    
    # Save training history
    with open(os.path.join(output_folder, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training configuration
    config = {
        'input_folder': args.input_folder,
        'epochs': args.epochs,
        'hidden_sizes': args.hidden_sizes,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'test_split': args.test_split,
        'best_validation_accuracy': best_val_acc,
        'num_classes': num_classes,
        'feature_dimension': input_size,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(os.path.join(output_folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Output folder: {output_folder}")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved: {os.path.join(output_folder, 'best_model.pth')}")
    print(f"📊 Training history: {os.path.join(output_folder, 'training_history.json')}")
    print(f"⚙️  Configuration: {os.path.join(output_folder, 'config.json')}")


if __name__ == "__main__":
    main()
