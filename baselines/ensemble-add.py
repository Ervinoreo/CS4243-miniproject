#!/usr/bin/env python3
"""
Ensemble Model for CAPTCHA Recognition
This script implements an ensemble of three character-level models (CNN, ResNet50, VGG16) 
and uses weighted averaging for final prediction aggregation.
"""

import os
import cv2
import numpy as np
import string
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

from tqdm import tqdm
import warnings
from collections import defaultdict
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CharacterDataProcessor:
    """Process individual character images for training"""
    
    def __init__(self, img_width_cnn=32, img_height_cnn=32, img_width_large=224, img_height_large=224):
        self.img_width_cnn = img_width_cnn
        self.img_height_cnn = img_height_cnn
        self.img_width_large = img_width_large
        self.img_height_large = img_height_large
        self.characters = string.digits + string.ascii_lowercase  # 0-9, a-z
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.characters)
        
    def parse_filename(self, filename):
        """Parse filename to extract CAPTCHA ID and character position"""
        basename = filename.replace('.png', '')
        parts = basename.split('_')
        if len(parts) >= 2:
            captcha_id = parts[0]
            position = int(parts[1]) if parts[1].isdigit() else 0
            return captcha_id, position
        return None, None
    
    def preprocess_image_cnn(self, image_path):
        """Preprocess image for CNN model (32x32, grayscale)"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.img_width_cnn, self.img_height_cnn))
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def preprocess_image_large(self, image_path):
        """Preprocess image for ResNet50/VGG16 models (224x224, RGB)"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_width_large, self.img_height_large))
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1))  # CHW format
        return normalized
    
    def load_character_data(self, data_dir, split_validation=True):
        """Load character images organized by class folders"""
        data_dir = Path(data_dir)
        
        images_cnn = []
        images_large = []
        labels = []
        image_paths = []
        
        print(f"Loading character data from {data_dir}...")
        
        # Process each character class folder
        for char_folder in sorted(data_dir.iterdir()):
            if not char_folder.is_dir() or char_folder.name == 'skipped_folders.txt':
                continue
                
            char = char_folder.name
            if char not in self.char_to_idx:
                print(f"Skipping unknown character: {char}")
                continue
                
            label = self.char_to_idx[char]
            
            # Process all images in this character folder
            for img_path in char_folder.glob('*.png'):
                # Preprocess for CNN
                img_cnn = self.preprocess_image_cnn(img_path)
                # Preprocess for large models
                img_large = self.preprocess_image_large(img_path)
                
                if img_cnn is not None and img_large is not None:
                    images_cnn.append(img_cnn)
                    images_large.append(img_large)
                    labels.append(label)
                    image_paths.append(str(img_path))
        
        print(f"Loaded {len(images_cnn)} character images")
        print(f"Character distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            char = self.idx_to_char[label_idx]
            print(f"  {char}: {count}")
        
        if split_validation:
            # Split into train and validation sets
            X_train_cnn, X_val_cnn, X_train_large, X_val_large, y_train, y_val, paths_train, paths_val = train_test_split(
                images_cnn, images_large, labels, image_paths, test_size=0.2, random_state=42, stratify=labels
            )
            return X_train_cnn, X_val_cnn, X_train_large, X_val_large, y_train, y_val, paths_train, paths_val
        else:
            return images_cnn, images_large, labels, image_paths

class CharacterDataset(Dataset):
    """PyTorch Dataset for character images"""
    
    def __init__(self, images, labels, image_paths=None, transform=None):
        self.images = images
        self.labels = labels
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # If transform is provided we expect the model to be a large pretrained model
            # Ensure image is in CHW tensor format and apply the transform (e.g., Normalize)
            if len(image.shape) == 2:  # Grayscale -> replicate channels
                image = np.stack([image] * 3, axis=0)
            # image is already CHW for large models from preprocess_image_large
            image = torch.FloatTensor(image)
            # Apply transform (should be a Compose containing Normalize)
            try:
                image = self.transform(image)
            except Exception:
                # If transform expects PIL or different input, fall back to using the tensor as-is
                pass
        else:
            if len(image.shape) == 2:  # Grayscale for CNN
                image = torch.FloatTensor(image).unsqueeze(0)
            else:  # RGB for large models
                image = torch.FloatTensor(image)
        
        label = torch.LongTensor([label])[0]
        return image, label

def simple_collate_fn(batch):
    """Simple collate function"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

class CharacterCNN(nn.Module):
    """CNN Model for Individual Character Classification"""
    
    def __init__(self, num_classes=36, img_size=32):
        super(CharacterCNN, self).__init__()
        self.num_classes = num_classes
        
        # CNN Feature Extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Calculate feature map size after convolutions
        feature_size = (img_size // 16) * (img_size // 16) * 256
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class CharacterResNet50(nn.Module):
    """ResNet50-based Model for Individual Character Classification"""
    
    def __init__(self, num_classes=36, pretrained=True):
        super(CharacterResNet50, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

class CharacterVGG16(nn.Module):
    """VGG16-based Model for Individual Character Classification"""
    
    def __init__(self, num_classes=36, pretrained=True):
        super(CharacterVGG16, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Get the number of features from VGG16's classifier
        num_features = self.vgg.classifier[0].in_features
        
        # Replace the classifier
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.vgg(x)

class ModelTrainer:
    """Training class for individual models"""
    
    def __init__(self, model, device='cpu', model_name='model'):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()
        
        if 'resnet' in model_name.lower() or 'vgg' in model_name.lower():
            # Different learning rates for pretrained models
            pretrained_params = []
            new_params = []
            
            for name, param in model.named_parameters():
                if 'fc' in name or 'classifier' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
            
            self.optimizer = optim.Adam([
                {'params': pretrained_params, 'lr': 0.0001},
                {'params': new_params, 'lr': 0.001}
            ], weight_decay=1e-4)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Training {self.model_name}"):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'best_{self.model_name}_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def predict_with_probabilities(self, dataloader):
        """Get predictions and probabilities"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                images, _ = batch
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)

class EnsembleModel:
    """Ensemble model combining CNN, ResNet50, and VGG16 with weighted averaging"""
    
    def __init__(self, processor, device='cpu'):
        self.processor = processor
        self.device = device
        
        # Initialize models
        self.cnn_model = CharacterCNN(num_classes=processor.num_classes)
        self.resnet_model = CharacterResNet50(num_classes=processor.num_classes)
        self.vgg_model = CharacterVGG16(num_classes=processor.num_classes)
        
        # Initialize trainers
        self.cnn_trainer = ModelTrainer(self.cnn_model, device, 'cnn')
        self.resnet_trainer = ModelTrainer(self.resnet_model, device, 'resnet50')
        self.vgg_trainer = ModelTrainer(self.vgg_model, device, 'vgg16')
        
        # Direct ensemble method: test all averaging strategies simultaneously
        self.ensemble_methods = ['simple_average', 'weighted_average', 'learned_weights']
        
        # Initialize weights for different averaging strategies
        self.all_weights = {
            'simple_average': np.array([1/3, 1/3, 1/3]),
            'weighted_average': np.array([1.0, 1.0, 1.0]) / 3.0,  # Will be updated based on accuracies
            'learned_weights': np.array([1.0, 1.0, 1.0]) / 3.0    # Will be optimized
        }
        
        # Transforms for different models
        self.transform_resnet_vgg = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train_individual_models(self, X_train_cnn, X_train_large, y_train, 
                               X_val_cnn, X_val_large, y_val, epochs=30):
        """Train all individual models"""
        print("Training individual models...")
        
        # CNN training
        print("\n=== Training CNN Model ===")
        train_dataset_cnn = CharacterDataset(X_train_cnn, y_train)
        val_dataset_cnn = CharacterDataset(X_val_cnn, y_val)
        train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True, collate_fn=simple_collate_fn)
        val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=64, shuffle=False, collate_fn=simple_collate_fn)
        
        history_cnn = self.cnn_trainer.train(train_loader_cnn, val_loader_cnn, epochs)
        
        # ResNet50 training
        print("\n=== Training ResNet50 Model ===")
        train_dataset_resnet = CharacterDataset(X_train_large, y_train, transform=self.transform_resnet_vgg)
        val_dataset_resnet = CharacterDataset(X_val_large, y_val, transform=self.transform_resnet_vgg)
        train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=32, shuffle=True, collate_fn=simple_collate_fn)
        val_loader_resnet = DataLoader(val_dataset_resnet, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)
        
        history_resnet = self.resnet_trainer.train(train_loader_resnet, val_loader_resnet, epochs)
        
        # VGG16 training
        print("\n=== Training VGG16 Model ===")
        train_dataset_vgg = CharacterDataset(X_train_large, y_train, transform=self.transform_resnet_vgg)
        val_dataset_vgg = CharacterDataset(X_val_large, y_val, transform=self.transform_resnet_vgg)
        train_loader_vgg = DataLoader(train_dataset_vgg, batch_size=32, shuffle=True, collate_fn=simple_collate_fn)
        val_loader_vgg = DataLoader(val_dataset_vgg, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)
        
        history_vgg = self.vgg_trainer.train(train_loader_vgg, val_loader_vgg, epochs)
        
        return history_cnn, history_resnet, history_vgg
    
    def train_ensemble_weights(self, X_val_cnn, X_val_large, y_val):
        """Train/learn ensemble weights for all methods using validation set predictions"""
        print(f"\n=== Learning Ensemble Weights (All Methods) ===")
        
        # Load best models
        self.cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
        self.resnet_model.load_state_dict(torch.load('best_resnet50_model.pth'))
        self.vgg_model.load_state_dict(torch.load('best_vgg16_model.pth'))
        
        # Get predictions from all models on validation set
        val_dataset_cnn = CharacterDataset(X_val_cnn, y_val)
        val_dataset_large = CharacterDataset(X_val_large, y_val, transform=self.transform_resnet_vgg)
        val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=64, shuffle=False, collate_fn=simple_collate_fn)
        val_loader_large = DataLoader(val_dataset_large, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)
        
        # Get predictions and probabilities
        cnn_preds, cnn_probs = self.cnn_trainer.predict_with_probabilities(val_loader_cnn)
        resnet_preds, resnet_probs = self.resnet_trainer.predict_with_probabilities(val_loader_large)
        vgg_preds, vgg_probs = self.vgg_trainer.predict_with_probabilities(val_loader_large)
        
        # Calculate individual model accuracies for weight initialization
        cnn_acc = accuracy_score(y_val, cnn_preds)
        resnet_acc = accuracy_score(y_val, resnet_preds)
        vgg_acc = accuracy_score(y_val, vgg_preds)
        
        print(f"Individual validation accuracies:")
        print(f"  CNN: {cnn_acc:.4f}")
        print(f"  ResNet50: {resnet_acc:.4f}")
        print(f"  VGG16: {vgg_acc:.4f}")
        
        # Method 1: Simple average (equal weights)
        self.all_weights['simple_average'] = np.array([1/3, 1/3, 1/3])
        
        # Method 2: Weighted average (accuracy-based weights)
        accs = np.array([cnn_acc, resnet_acc, vgg_acc])
        self.all_weights['weighted_average'] = accs / np.sum(accs)
        
        # Method 3: Learned weights (optimization-based)
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_probs = (weights[0] * cnn_probs + 
                            weights[1] * resnet_probs + 
                            weights[2] * vgg_probs)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            return -accuracy_score(y_val, ensemble_preds)  # Negative for minimization
        
        # Optimize weights
        initial_weights = np.array([cnn_acc, resnet_acc, vgg_acc])
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            self.all_weights['learned_weights'] = result.x / np.sum(result.x)
        else:
            print("Weight optimization failed, using accuracy-based weights for learned method")
            self.all_weights['learned_weights'] = self.all_weights['weighted_average'].copy()
        
        # Print all weights
        print(f"\nEnsemble weights computed:")
        for method, weights in self.all_weights.items():
            print(f"  {method}: {weights}")
        
        # Test all ensemble methods on validation set
        val_results = {}
        for method, weights in self.all_weights.items():
            ensemble_probs = (weights[0] * cnn_probs + 
                             weights[1] * resnet_probs + 
                             weights[2] * vgg_probs)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            ensemble_acc = accuracy_score(y_val, ensemble_preds)
            val_results[method] = ensemble_acc
            
        print(f"\nValidation accuracies:")
        for method, acc in val_results.items():
            improvement = acc - max(cnn_acc, resnet_acc, vgg_acc)
            print(f"  {method}: {acc:.4f} (improvement: {improvement:+.4f})")
        
        best_individual = max(cnn_acc, resnet_acc, vgg_acc)
        print(f"  Best individual: {best_individual:.4f}")
        
        # Save all ensemble weights
        np.save('ensemble_weights_all.npy', self.all_weights)
        print(f"\nAll ensemble weights saved to ensemble_weights_all.npy")
        
        return {
            'cnn_acc': cnn_acc,
            'resnet_acc': resnet_acc, 
            'vgg_acc': vgg_acc,
            'val_results': val_results,
            'all_weights': self.all_weights
        }
    
    def predict_ensemble(self, X_test_cnn, X_test_large):
        """Make ensemble predictions using all weighted averaging methods"""
        # Load best models
        self.cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
        self.resnet_model.load_state_dict(torch.load('best_resnet50_model.pth'))
        self.vgg_model.load_state_dict(torch.load('best_vgg16_model.pth'))
        
        # Load ensemble weights
        try:
            self.all_weights = np.load('ensemble_weights_all.npy', allow_pickle=True).item()
        except FileNotFoundError:
            print("Warning: ensemble weights not found, using default weights")
            self.all_weights = {
                'simple_average': np.array([1/3, 1/3, 1/3]),
                'weighted_average': np.array([1/3, 1/3, 1/3]),
                'learned_weights': np.array([1/3, 1/3, 1/3])
            }
        
        # Create test datasets and loaders
        test_dataset_cnn = CharacterDataset(X_test_cnn, [0] * len(X_test_cnn))  # Dummy labels
        test_dataset_large = CharacterDataset(X_test_large, [0] * len(X_test_large), transform=self.transform_resnet_vgg)
        test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64, shuffle=False, collate_fn=simple_collate_fn)
        test_loader_large = DataLoader(test_dataset_large, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)
        
        # Get predictions from all models
        cnn_preds, cnn_probs = self.cnn_trainer.predict_with_probabilities(test_loader_cnn)
        resnet_preds, resnet_probs = self.resnet_trainer.predict_with_probabilities(test_loader_large)
        vgg_preds, vgg_probs = self.vgg_trainer.predict_with_probabilities(test_loader_large)
        
        # Ensemble predictions using all averaging methods
        ensemble_results = {}
        for method, weights in self.all_weights.items():
            ensemble_probs = (weights[0] * cnn_probs + 
                             weights[1] * resnet_probs + 
                             weights[2] * vgg_probs)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            ensemble_results[method] = {
                'predictions': ensemble_preds,
                'probabilities': ensemble_probs,
                'weights': weights
            }
        
        return ensemble_results, {
            'cnn': (cnn_preds, cnn_probs),
            'resnet': (resnet_preds, resnet_probs),
            'vgg': (vgg_preds, vgg_probs)
        }
    
    def aggregate_to_captcha_predictions(self, char_predictions, image_paths):
        """Aggregate character predictions to CAPTCHA-level predictions"""
        captcha_predictions = defaultdict(list)
        
        # Group predictions by CAPTCHA ID
        for pred, img_path in zip(char_predictions, image_paths):
            filename = Path(img_path).name
            captcha_id, position = self.processor.parse_filename(filename)
            
            if captcha_id is not None and position is not None:
                captcha_predictions[captcha_id].append((position, pred))
        
        # Sort by position and create final CAPTCHA predictions
        final_predictions = {}
        true_captchas = {}
        
        for captcha_id, char_data in captcha_predictions.items():
            # Sort by position
            char_data.sort(key=lambda x: x[0])
            
            # Create predicted CAPTCHA string
            pred_captcha = ''.join([self.processor.idx_to_char[pred] for _, pred in char_data])
            final_predictions[captcha_id] = pred_captcha
            
            # Extract true CAPTCHA from ID
            true_captchas[captcha_id] = captcha_id
        
        return final_predictions, true_captchas
    
    def evaluate_comprehensive(self, X_test_cnn, X_test_large, y_test, test_paths):
        """Comprehensive evaluation with both character and CAPTCHA level metrics for all ensemble methods"""
        # Get ensemble predictions for all methods
        ensemble_results, individual_preds = self.predict_ensemble(X_test_cnn, X_test_large)
        
        # Evaluate each ensemble method
        all_results = {}
        
        for method, ensemble_data in ensemble_results.items():
            ensemble_preds = ensemble_data['predictions']
            
            # Character-level evaluation
            char_accuracy = accuracy_score(y_test, ensemble_preds)
            
            # Calculate precision, recall, F1 for character level
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_test, ensemble_preds, average='weighted', zero_division=0
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_test, ensemble_preds, average='macro', zero_division=0
            )
            
            # CAPTCHA-level evaluation
            captcha_predictions, true_captchas = self.aggregate_to_captcha_predictions(ensemble_preds, test_paths)
            
            captcha_correct = 0
            captcha_total = 0
            for captcha_id in captcha_predictions:
                if captcha_id in true_captchas:
                    if captcha_predictions[captcha_id] == true_captchas[captcha_id]:
                        captcha_correct += 1
                    captcha_total += 1
            
            captcha_accuracy = captcha_correct / captcha_total if captcha_total > 0 else 0
            
            all_results[method] = {
                'char_accuracy': char_accuracy,
                'char_correct': np.sum(ensemble_preds == y_test),
                'char_total': len(y_test),
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'captcha_accuracy': captcha_accuracy,
                'captcha_correct': captcha_correct,
                'captcha_total': captcha_total,
                'captcha_predictions': captcha_predictions,
                'true_captchas': true_captchas,
                'char_predictions': ensemble_preds,
                'weights': ensemble_data['weights']
            }
        
        # Individual model evaluations
        individual_results = {}
        for model_name, (preds, probs) in individual_preds.items():
            acc = accuracy_score(y_test, preds)
            individual_results[model_name] = acc
        
        # Add individual results to all_results
        all_results['individual_results'] = individual_results
        
        return all_results

def visualize_results(results, histories):
    """Visualize training and evaluation results for all ensemble methods"""
    # Plot training histories
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    model_names = ['CNN', 'ResNet50', 'VGG16']
    for i, (history, name) in enumerate(zip(histories, model_names)):
        # Loss plot
        axes[0, i].plot(history['train_loss'], label='Training Loss')
        axes[0, i].plot(history['val_loss'], label='Validation Loss')
        axes[0, i].set_title(f'{name} - Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()
        axes[0, i].grid(True)
        
        # Accuracy plot
        axes[1, i].plot(history['train_acc'], label='Training Accuracy')
        axes[1, i].plot(history['val_acc'], label='Validation Accuracy')
        axes[1, i].set_title(f'{name} - Accuracy')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig('./ensemble_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot model comparison for all ensemble methods
    plt.figure(figsize=(15, 10))
    
    # Individual model accuracies + all ensemble methods
    models = ['CNN', 'ResNet50', 'VGG16', 'Simple Avg', 'Weighted Avg', 'Learned Weights']
    accuracies = [
        results['individual_results']['cnn'],
        results['individual_results']['resnet'],
        results['individual_results']['vgg'],
        results['simple_average']['char_accuracy'],
        results['weighted_average']['char_accuracy'],
        results['learned_weights']['char_accuracy']
    ]
    
    plt.subplot(2, 2, 1)
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orange', 'lightcoral']
    bars = plt.bar(models, accuracies, color=colors)
    plt.title('Character-Level Accuracy Comparison (All Methods)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # CAPTCHA accuracy comparison
    plt.subplot(2, 2, 2)
    captcha_methods = ['Simple Avg', 'Weighted Avg', 'Learned Weights']
    captcha_accs = [
        results['simple_average']['captcha_accuracy'],
        results['weighted_average']['captcha_accuracy'],
        results['learned_weights']['captcha_accuracy']
    ]
    bars = plt.bar(captcha_methods, captcha_accs, color=['gold', 'orange', 'lightcoral'])
    plt.title('CAPTCHA-Level Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, max(captcha_accs) * 1.2)
    plt.xticks(rotation=45)
    
    for bar, acc in zip(bars, captcha_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Improvement comparison
    plt.subplot(2, 2, 3)
    best_individual = max(results['individual_results'].values())
    improvements = [
        results['simple_average']['char_accuracy'] - best_individual,
        results['weighted_average']['char_accuracy'] - best_individual,
        results['learned_weights']['char_accuracy'] - best_individual
    ]
    ensemble_methods = ['Simple Avg', 'Weighted Avg', 'Learned Weights']
    colors = ['gold', 'orange', 'lightcoral']
    bars = plt.bar(ensemble_methods, improvements, color=colors)
    plt.title('Improvement over Best Individual Model')
    plt.ylabel('Accuracy Improvement')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{imp:+.3f}', ha='center', va='bottom' if imp >= 0 else 'top')
    
    # Weights visualization
    plt.subplot(2, 2, 4)
    x = np.arange(3)
    width = 0.25
    
    simple_weights = results['simple_average']['weights']
    weighted_weights = results['weighted_average']['weights']
    learned_weights = results['learned_weights']['weights']
    
    plt.bar(x - width, simple_weights, width, label='Simple Avg', color='gold', alpha=0.8)
    plt.bar(x, weighted_weights, width, label='Weighted Avg', color='orange', alpha=0.8)
    plt.bar(x + width, learned_weights, width, label='Learned Weights', color='lightcoral', alpha=0.8)
    
    plt.title('Ensemble Weights Comparison')
    plt.ylabel('Weight')
    plt.xlabel('Model')
    plt.xticks(x, ['CNN', 'ResNet50', 'VGG16'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./ensemble_model_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved:")
    print("  - ensemble_training_history.png")
    print("  - ensemble_model_comparison_all.png")

def main():
    """Main function"""
    # Configuration
    TRAIN_DIR = "./char_dataset/labeled_train"
    TEST_DIR = "./char_dataset/labeled_test"
    EPOCHS = 100
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=== Ensemble Model for CAPTCHA Recognition ===")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    
    # Initialize data processor
    processor = CharacterDataProcessor()
    
    # Load training data
    print("\nLoading training data...")
    X_train_cnn, X_val_cnn, X_train_large, X_val_large, y_train, y_val, paths_train, paths_val = processor.load_character_data(
        TRAIN_DIR, split_validation=True
    )
    
    print(f"Training set: {len(X_train_cnn)} character images")
    print(f"Validation set: {len(X_val_cnn)} character images")
    
    # Load test data
    print("\nLoading test data...")
    X_test_cnn, X_test_large, y_test, paths_test = processor.load_character_data(TEST_DIR, split_validation=False)
    print(f"Test set: {len(X_test_cnn)} character images")
    
    # Initialize ensemble model
    ensemble = EnsembleModel(processor, device)
    
    # Train individual models
    history_cnn, history_resnet, history_vgg = ensemble.train_individual_models(
        X_train_cnn, X_train_large, y_train, X_val_cnn, X_val_large, y_val, epochs=EPOCHS
    )
    
    # Train ensemble weights
    ensemble.train_ensemble_weights(X_val_cnn, X_val_large, y_val)
    
    # Evaluate on test set
    print("\n=== Evaluating Ensemble Model ===")
    results = ensemble.evaluate_comprehensive(X_test_cnn, X_test_large, y_test, paths_test)
    
    # Print results for all ensemble methods
    print(f"\n{'='*60}")
    print(f"ENSEMBLE MODEL TEST RESULTS (ALL METHODS)")
    print(f"{'='*60}")
    
    print(f"\nIndividual Model Character-Level Accuracies:")
    print(f"  CNN: {results['individual_results']['cnn']:.4f}")
    print(f"  ResNet50: {results['individual_results']['resnet']:.4f}")
    print(f"  VGG16: {results['individual_results']['vgg']:.4f}")
    
    best_individual = max(results['individual_results'].values())
    
    # Print results for each ensemble method
    for method in ['simple_average', 'weighted_average', 'learned_weights']:
        if method in results:
            method_results = results[method]
            improvement = method_results['char_accuracy'] - best_individual
            
            print(f"\n=== {method.upper().replace('_', ' ')} ENSEMBLE ===")
            print(f"Weights: {method_results['weights']}")
            print(f"Character-Level Metrics:")
            print(f"  Accuracy: {method_results['char_accuracy']:.4f} ({method_results['char_correct']}/{method_results['char_total']})")
            print(f"  Precision (weighted): {method_results['precision_weighted']:.4f}")
            print(f"  Recall (weighted): {method_results['recall_weighted']:.4f}")
            print(f"  F1-Score (weighted): {method_results['f1_weighted']:.4f}")
            print(f"CAPTCHA-Level Metrics:")
            print(f"  Accuracy: {method_results['captcha_accuracy']:.4f} ({method_results['captcha_correct']}/{method_results['captcha_total']})")
            print(f"Improvement over best individual: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Find best ensemble method
    ensemble_methods = ['simple_average', 'weighted_average', 'learned_weights']
    best_ensemble_method = max(ensemble_methods, key=lambda m: results[m]['char_accuracy'] if m in results else 0)
    best_ensemble_acc = results[best_ensemble_method]['char_accuracy']
    
    print(f"\n=== SUMMARY ===")
    print(f"Best individual model: {best_individual:.4f}")
    print(f"Best ensemble method: {best_ensemble_method} ({best_ensemble_acc:.4f})")
    print(f"Overall improvement: {best_ensemble_acc - best_individual:+.4f} ({(best_ensemble_acc - best_individual)*100:+.2f}%)")
    
    # Show sample CAPTCHA predictions for best ensemble method
    print(f"\nSample CAPTCHA Predictions (Best Method: {best_ensemble_method}):")
    captcha_predictions = results[best_ensemble_method]['captcha_predictions']
    true_captchas = results[best_ensemble_method]['true_captchas']
    
    sample_count = 0
    for captcha_id in list(captcha_predictions.keys())[:10]:
        if captcha_id in true_captchas:
            pred = captcha_predictions[captcha_id]
            true = true_captchas[captcha_id]
            status = "✓" if pred == true else "✗"
            print(f"  {status} True: {true}, Predicted: {pred}")
            sample_count += 1
    
    # Save detailed results for all methods
    print("\nSaving results...")
    
    # Save metrics to text file
    with open('./ensemble_test_results_all.txt', 'w') as f:
        f.write("ENSEMBLE MODEL TEST RESULTS (ALL METHODS)\n")
        f.write("="*50 + "\n\n")
        
        f.write("Individual Model Character-Level Accuracies:\n")
        f.write(f"  CNN: {results['individual_results']['cnn']:.4f}\n")
        f.write(f"  ResNet50: {results['individual_results']['resnet']:.4f}\n")
        f.write(f"  VGG16: {results['individual_results']['vgg']:.4f}\n\n")
        
        best_individual = max(results['individual_results'].values())
        
        # Write results for each ensemble method
        for method in ['simple_average', 'weighted_average', 'learned_weights']:
            if method in results:
                method_results = results[method]
                improvement = method_results['char_accuracy'] - best_individual
                
                f.write(f"=== {method.upper().replace('_', ' ')} ENSEMBLE ===\n")
                f.write(f"Weights: {method_results['weights']}\n")
                f.write(f"Character-Level Metrics:\n")
                f.write(f"  Accuracy: {method_results['char_accuracy']:.4f} ({method_results['char_correct']}/{method_results['char_total']})\n")
                f.write(f"  Precision (weighted): {method_results['precision_weighted']:.4f}\n")
                f.write(f"  Recall (weighted): {method_results['recall_weighted']:.4f}\n")
                f.write(f"  F1-Score (weighted): {method_results['f1_weighted']:.4f}\n")
                f.write(f"CAPTCHA-Level Metrics:\n")
                f.write(f"  Accuracy: {method_results['captcha_accuracy']:.4f} ({method_results['captcha_correct']}/{method_results['captcha_total']})\n")
                f.write(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)\n\n")
        
        # Summary
        ensemble_methods = ['simple_average', 'weighted_average', 'learned_weights']
        best_ensemble_method = max(ensemble_methods, key=lambda m: results[m]['char_accuracy'] if m in results else 0)
        best_ensemble_acc = results[best_ensemble_method]['char_accuracy']
        
        f.write("=== SUMMARY ===\n")
        f.write(f"Best individual model: {best_individual:.4f}\n")
        f.write(f"Best ensemble method: {best_ensemble_method} ({best_ensemble_acc:.4f})\n")
        f.write(f"Overall improvement: {best_ensemble_acc - best_individual:+.4f} ({(best_ensemble_acc - best_individual)*100:+.2f}%)\n")
    
    # Save detailed results as pickle
    with open('./ensemble_test_results_all.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Visualize results
    histories = [history_cnn, history_resnet, history_vgg]
    visualize_results(results, histories)
    
    print("\nFiles saved:")
    print("  - ensemble_test_results_all.txt (human-readable results)")
    print("  - ensemble_test_results_all.pkl (detailed results for analysis)")
    print("  - best_cnn_model.pth")
    print("  - best_resnet50_model.pth") 
    print("  - best_vgg16_model.pth")
    print("  - ensemble_weights_all.npy")

if __name__ == "__main__":
    main()
