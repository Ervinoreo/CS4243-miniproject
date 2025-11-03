#!/usr/bin/env python3
"""
Ensemble Model for CAPTCHA Recognition
This script implements an ensemble of three character-level models (CNN, ResNet50, VGG16) 
and uses a decision tree for final prediction aggregation.
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
import joblib

from tqdm import tqdm
import warnings
from collections import defaultdict

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
            # Convert to PIL Image for transforms
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=0)  # Convert to 3-channel
            image = torch.FloatTensor(image)
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
    """Ensemble model combining CNN, ResNet50, and VGG16 with decision tree"""
    
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
        
        # Meta-learner (Decision Tree)
        self.meta_learner = DecisionTreeClassifier(random_state=42, max_depth=10)
        
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
    
    def train_meta_learner(self, X_val_cnn, X_val_large, y_val):
        """Train meta-learner using validation set predictions"""
        print("\n=== Training Meta-Learner (Decision Tree) ===")
        
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
        
        # Create feature matrix for meta-learner
        # Features: predictions + top probabilities from each model
        meta_features = np.column_stack([
            cnn_preds, np.max(cnn_probs, axis=1),
            resnet_preds, np.max(resnet_probs, axis=1),
            vgg_preds, np.max(vgg_probs, axis=1)
        ])
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y_val)
        
        # Save meta-learner
        joblib.dump(self.meta_learner, 'meta_learner.pkl')
        
        print(f"Meta-learner trained with {len(meta_features)} samples")
    
    def predict_ensemble(self, X_test_cnn, X_test_large):
        """Make ensemble predictions"""
        # Load best models
        self.cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
        self.resnet_model.load_state_dict(torch.load('best_resnet50_model.pth'))
        self.vgg_model.load_state_dict(torch.load('best_vgg16_model.pth'))
        
        # Load meta-learner
        self.meta_learner = joblib.load('meta_learner.pkl')
        
        # Create test datasets and loaders
        test_dataset_cnn = CharacterDataset(X_test_cnn, [0] * len(X_test_cnn))  # Dummy labels
        test_dataset_large = CharacterDataset(X_test_large, [0] * len(X_test_large), transform=self.transform_resnet_vgg)
        test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64, shuffle=False, collate_fn=simple_collate_fn)
        test_loader_large = DataLoader(test_dataset_large, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)
        
        # Get predictions from all models
        cnn_preds, cnn_probs = self.cnn_trainer.predict_with_probabilities(test_loader_cnn)
        resnet_preds, resnet_probs = self.resnet_trainer.predict_with_probabilities(test_loader_large)
        vgg_preds, vgg_probs = self.vgg_trainer.predict_with_probabilities(test_loader_large)
        
        # Create feature matrix for meta-learner
        meta_features = np.column_stack([
            cnn_preds, np.max(cnn_probs, axis=1),
            resnet_preds, np.max(resnet_probs, axis=1),
            vgg_preds, np.max(vgg_probs, axis=1)
        ])
        
        # Get ensemble predictions
        ensemble_preds = self.meta_learner.predict(meta_features)
        
        return ensemble_preds, {
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
        """Comprehensive evaluation with both character and CAPTCHA level metrics"""
        # Get ensemble predictions
        ensemble_preds, individual_preds = self.predict_ensemble(X_test_cnn, X_test_large)
        
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
        
        # Individual model evaluations
        individual_results = {}
        for model_name, (preds, probs) in individual_preds.items():
            acc = accuracy_score(y_test, preds)
            individual_results[model_name] = acc
        
        return {
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
            'individual_results': individual_results
        }

def visualize_results(results, histories):
    """Visualize training and evaluation results"""
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
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    
    # Individual model accuracies
    models = ['CNN', 'ResNet50', 'VGG16', 'Ensemble']
    accuracies = [
        results['individual_results']['cnn'],
        results['individual_results']['resnet'],
        results['individual_results']['vgg'],
        results['char_accuracy']
    ]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    plt.title('Character-Level Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # CAPTCHA vs Character accuracy
    plt.subplot(1, 2, 2)
    metrics = ['Character Accuracy', 'CAPTCHA Accuracy']
    ensemble_scores = [results['char_accuracy'], results['captcha_accuracy']]
    bars = plt.bar(metrics, ensemble_scores, color=['gold', 'orange'])
    plt.title('Ensemble Model Performance')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for bar, score in zip(bars, ensemble_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./ensemble_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved:")
    print("  - ensemble_training_history.png")
    print("  - ensemble_model_comparison.png")

def main():
    """Main function"""
    # Configuration
    TRAIN_DIR = "./char_dataset/labeled_train"
    TEST_DIR = "./char_dataset/labeled_test"
    EPOCHS = 30
    
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
    
    # Train meta-learner
    ensemble.train_meta_learner(X_val_cnn, X_val_large, y_val)
    
    # Evaluate on test set
    print("\n=== Evaluating Ensemble Model ===")
    results = ensemble.evaluate_comprehensive(X_test_cnn, X_test_large, y_test, paths_test)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"ENSEMBLE MODEL TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\nIndividual Model Character-Level Accuracies:")
    print(f"  CNN: {results['individual_results']['cnn']:.4f}")
    print(f"  ResNet50: {results['individual_results']['resnet']:.4f}")
    print(f"  VGG16: {results['individual_results']['vgg']:.4f}")
    
    print(f"\nEnsemble Character-Level Metrics:")
    print(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {results['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro): {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro): {results['f1_macro']:.4f}")
    
    print(f"\nEnsemble CAPTCHA-Level Metrics:")
    print(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})")
    
    # Show improvement
    best_individual = max(results['individual_results'].values())
    improvement = results['char_accuracy'] - best_individual
    print(f"\nEnsemble Improvement:")
    print(f"  Best individual model: {best_individual:.4f}")
    print(f"  Ensemble model: {results['char_accuracy']:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    # Show sample CAPTCHA predictions
    print(f"\nSample CAPTCHA Predictions:")
    captcha_predictions = results['captcha_predictions']
    true_captchas = results['true_captchas']
    
    sample_count = 0
    for captcha_id in list(captcha_predictions.keys())[:10]:
        if captcha_id in true_captchas:
            pred = captcha_predictions[captcha_id]
            true = true_captchas[captcha_id]
            status = "✓" if pred == true else "✗"
            print(f"  {status} True: {true}, Predicted: {pred}")
            sample_count += 1
    
    # Save detailed results
    print("\nSaving results...")
    
    # Save metrics to text file
    with open('./ensemble_test_results.txt', 'w') as f:
        f.write("ENSEMBLE MODEL TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Individual Model Character-Level Accuracies:\n")
        f.write(f"  CNN: {results['individual_results']['cnn']:.4f}\n")
        f.write(f"  ResNet50: {results['individual_results']['resnet']:.4f}\n")
        f.write(f"  VGG16: {results['individual_results']['vgg']:.4f}\n\n")
        
        f.write("Ensemble Character-Level Metrics:\n")
        f.write(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})\n")
        f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {results['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
        f.write(f"  Recall (macro): {results['recall_macro']:.4f}\n")
        f.write(f"  F1-Score (macro): {results['f1_macro']:.4f}\n\n")
        
        f.write("Ensemble CAPTCHA-Level Metrics:\n")
        f.write(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})\n\n")
        
        f.write("Ensemble Improvement:\n")
        f.write(f"  Best individual model: {best_individual:.4f}\n")
        f.write(f"  Ensemble model: {results['char_accuracy']:.4f}\n")
        f.write(f"  Improvement: {improvement:.4f} ({improvement*100:.2f}%)\n")
    
    # Save detailed results as pickle
    with open('./ensemble_test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Visualize results
    histories = [history_cnn, history_resnet, history_vgg]
    visualize_results(results, histories)
    
    print("\nFiles saved:")
    print("  - ensemble_test_results.txt (human-readable results)")
    print("  - ensemble_test_results.pkl (detailed results for analysis)")
    print("  - best_cnn_model.pth")
    print("  - best_resnet50_model.pth") 
    print("  - best_vgg16_model.pth")
    print("  - meta_learner.pkl")

if __name__ == "__main__":
    main()
