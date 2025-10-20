#!/usr/bin/env python3
"""
Character-Level CNN Model for CAPTCHA Recognition
This script implements a CNN model to classify individual characters, then aggregates them for CAPTCHA-level evaluation.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import string
import re
from tqdm import tqdm
import pickle
from collections import defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CharacterDataProcessor:
    """Process individual character images for training"""
    
    def __init__(self, img_width=32, img_height=32):
        self.img_width = img_width
        self.img_height = img_height
        self.characters = string.digits + string.ascii_lowercase  # 0-9, a-z
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.characters)
        
    def parse_filename(self, filename):
        """Parse filename to extract CAPTCHA ID and character position
        Format: captcha_id_position.png (e.g., '002e23_000.png')
        """
        basename = filename.replace('.png', '')
        parts = basename.split('_')
        if len(parts) >= 2:
            captcha_id = parts[0]
            position = int(parts[1])
            return captcha_id, position
        return None, None
    
    def preprocess_image(self, image_path):
        """Preprocess a single character image"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        resized = cv2.resize(gray, (self.img_width, self.img_height))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def load_character_data(self, data_dir, split_validation=True):
        """Load character images organized by class folders"""
        data_dir = Path(data_dir)
        
        images = []
        labels = []
        metadata = []  # Store (captcha_id, position, true_char) for each sample
        
        print(f"Loading character data from {data_dir}...")
        
        # Process each character class folder
        for char_folder in sorted(data_dir.iterdir()):
            if not char_folder.is_dir() or char_folder.name == 'skipped_folders.txt':
                continue
                
            char_label = char_folder.name  # This is the true character (e.g., '0', 'a', etc.)
            if char_label not in self.char_to_idx:
                print(f"Warning: Unknown character class '{char_label}', skipping...")
                continue
                
            class_idx = self.char_to_idx[char_label]
            
            # Process all images in this character folder
            image_files = list(char_folder.glob("*.png"))
            print(f"Processing {len(image_files)} images for character '{char_label}'")
            
            for img_path in tqdm(image_files, desc=f"Loading '{char_label}'"):
                # Preprocess image
                processed_img = self.preprocess_image(img_path)
                if processed_img is None:
                    continue
                
                # Parse filename for metadata
                captcha_id, position = self.parse_filename(img_path.name)
                if captcha_id is None:
                    continue
                
                images.append(processed_img)
                labels.append(class_idx)
                # Store the actual character label (from folder name), not the captcha_id
                metadata.append((captcha_id, position, char_label))
        
        print(f"Loaded {len(images)} character images")
        print(f"Character distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            char = self.idx_to_char[label_idx]
            print(f"  '{char}': {count} samples")
        
        if split_validation:
            # Split into train and validation sets
            X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
                images, labels, metadata, test_size=0.2, random_state=42, stratify=labels
            )
            return X_train, X_val, y_train, y_val, meta_train, meta_val
        else:
            return images, labels, metadata

class CharacterDataset(Dataset):
    """PyTorch Dataset for individual character images"""
    
    def __init__(self, images, labels, metadata=None, transform=None):
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and add channel dimension
            image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        
        label = torch.LongTensor([label])[0]  # Convert to scalar tensor
        
        if self.metadata:
            return image, label, self.metadata[idx]
        else:
            return image, label

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
        # CNN Feature Extraction
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class CharacterTrainer:
    """Training class for Character CNN"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Handle both cases: with and without metadata
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
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
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle both cases: with and without metadata
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=100):
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_character_model.pth')
                print("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def predict_characters(self, dataloader, device='cpu'):
        """Predict individual characters"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:  # Has metadata
                    images, labels, metadata = batch
                    all_metadata.extend(metadata)
                else:
                    images, labels = batch
                
                images = images.to(device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_probabilities, all_metadata
    
    def aggregate_to_captcha_predictions(self, char_predictions, char_metadata, processor):
        """Aggregate character predictions to CAPTCHA-level predictions"""
        captcha_predictions = defaultdict(list)
        
        # Group predictions by CAPTCHA ID
        for pred, (captcha_id, position, true_char) in zip(char_predictions, char_metadata):
            predicted_char = processor.idx_to_char[pred]
            captcha_predictions[captcha_id].append((position, predicted_char, true_char))
        
        # Sort by position and create final CAPTCHA predictions
        final_predictions = {}
        true_captchas = {}
        
        for captcha_id, char_data in captcha_predictions.items():
            # Sort by position
            char_data.sort(key=lambda x: x[0])
            
            # Extract predicted and true sequences
            pred_sequence = ''.join([char_data[i][1] for i in range(len(char_data))])
            true_sequence = ''.join([char_data[i][2] for i in range(len(char_data))])
            
            final_predictions[captcha_id] = pred_sequence
            true_captchas[captcha_id] = true_sequence
        
        return final_predictions, true_captchas
    
    def evaluate_comprehensive(self, dataloader, processor, device='cpu'):
        """Comprehensive evaluation with both character and CAPTCHA level metrics"""
        char_predictions, char_probabilities, char_metadata = self.predict_characters(dataloader, device)
        
        # Character-level evaluation
        # The true character labels should come from the actual labels passed to the dataset, not metadata
        # We need to get them directly from the dataloader
        true_char_labels = []
        
        # Re-iterate through dataloader to get true labels
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                
                true_char_labels.extend(labels.cpu().numpy())
        
        char_accuracy = accuracy_score(true_char_labels, char_predictions)
        
        # Calculate precision, recall, F1 for character level
        precision, recall, f1, support = precision_recall_fscore_support(
            true_char_labels, char_predictions, average='weighted', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_char_labels, char_predictions, average='macro', zero_division=0
        )
        
        # CAPTCHA-level evaluation
        captcha_predictions, true_captchas = self.aggregate_to_captcha_predictions(
            char_predictions, char_metadata, processor
        )
        
        # Calculate CAPTCHA accuracy
        correct_captchas = 0
        total_captchas = len(captcha_predictions)
        
        for captcha_id in captcha_predictions:
            if captcha_predictions[captcha_id] == true_captchas[captcha_id]:
                correct_captchas += 1
        
        captcha_accuracy = correct_captchas / total_captchas if total_captchas > 0 else 0
        
        results = {
            'char_accuracy': char_accuracy,
            'char_correct': sum(p == t for p, t in zip(char_predictions, true_char_labels)),
            'char_total': len(char_predictions),
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'captcha_accuracy': captcha_accuracy,
            'captcha_correct': correct_captchas,
            'captcha_total': total_captchas,
            'char_predictions': char_predictions,
            'char_metadata': char_metadata,
            'captcha_predictions': captcha_predictions,
            'true_captchas': true_captchas
        }
        
        return results

def visualize_character_predictions(X_test, y_test, predictions, processor, num_samples=20):
    """Visualize character predictions"""
    plt.figure(figsize=(20, 4 * (num_samples // 5)))
    
    for i in range(min(num_samples, len(X_test))):
        plt.subplot(num_samples // 5, 5, i + 1)
        
        # Display image
        img = X_test[i].squeeze()
        plt.imshow(img, cmap='gray')
        
        # Get texts
        pred_char = processor.idx_to_char[predictions[i]]
        true_char = processor.idx_to_char[y_test[i]]
        
        # Set title
        color = 'green' if pred_char == true_char else 'red'
        plt.title(f'True: {true_char}, Pred: {pred_char}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./character_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Character predictions saved to './character_predictions.png'")

def main():
    """Main function"""
    # Configuration
    TRAIN_DIR = "./char_dataset/labeled_train"
    TEST_DIR = "./char_dataset/labeled_test"
    BATCH_SIZE = 64
    EPOCHS = 100
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=== Character-Level CNN Model (PyTorch) ===")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    
    # Initialize data processor
    processor = CharacterDataProcessor()
    
    # Load training data
    print("\nLoading training data...")
    X_train, X_val, y_train, y_val, meta_train, meta_val = processor.load_character_data(TRAIN_DIR, split_validation=True)
    
    print(f"Training set: {len(X_train)} character images")
    print(f"Validation set: {len(X_val)} character images")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test, meta_test = processor.load_character_data(TEST_DIR, split_validation=False)
    print(f"Test set: {len(X_test)} character images")
    
    # Create datasets and dataloaders
    train_dataset = CharacterDataset(X_train, y_train, meta_train)
    val_dataset = CharacterDataset(X_val, y_val, meta_val)
    test_dataset = CharacterDataset(X_test, y_test, meta_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model
    print("\nBuilding model...")
    model = CharacterCNN(num_classes=processor.num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer
    trainer = CharacterTrainer(model, device)
    
    print("\nTraining model...")
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_character_model.pth'))
    trainer.model = model.to(device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate_comprehensive(test_loader, processor, device)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    
    print(f"\nCharacter-Level Metrics:")
    print(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {results['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro): {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro): {results['f1_macro']:.4f}")
    
    print(f"\nCAPTCHA-Level Metrics (Aggregated):")
    print(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})")
    
    # Show some example CAPTCHA predictions
    print(f"\nSample CAPTCHA Predictions:")
    captcha_predictions = results['captcha_predictions']
    true_captchas = results['true_captchas']
    
    sample_count = 0
    for captcha_id in list(captcha_predictions.keys())[:10]:
        pred_text = captcha_predictions[captcha_id]
        true_text = true_captchas[captcha_id]
        status = "✓" if pred_text == true_text else "✗"
        print(f"  {status} ID: {captcha_id} | True: '{true_text}' | Pred: '{pred_text}'")
        sample_count += 1
        if sample_count >= 10:
            break
    
    # Visualize some character predictions
    print("\nVisualizing character predictions...")
    visualize_character_predictions(X_test, y_test, results['char_predictions'], processor, num_samples=20)
    
    # Save detailed results
    print("\nSaving results...")
    
    # Save metrics to text file
    with open('./test_results.txt', 'w') as f:
        f.write("Character-Level CNN Test Results\n")
        f.write("="*40 + "\n\n")
        
        f.write("Character-Level Metrics:\n")
        f.write(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})\n")
        f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {results['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
        f.write(f"  Recall (macro): {results['recall_macro']:.4f}\n")
        f.write(f"  F1-Score (macro): {results['f1_macro']:.4f}\n\n")
        
        f.write("CAPTCHA-Level Metrics (Aggregated):\n")
        f.write(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})\n\n")
        
        f.write("Sample CAPTCHA Predictions:\n")
        sample_count = 0
        for captcha_id in captcha_predictions:
            if sample_count >= 20:
                break
            pred_text = captcha_predictions[captcha_id]
            true_text = true_captchas[captcha_id]
            status = "✓" if pred_text == true_text else "✗"
            f.write(f"  {status} ID: {captcha_id} | True: '{true_text}' | Pred: '{pred_text}'\n")
            sample_count += 1
    
    # Save detailed results as pickle
    with open('./test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Files saved:")
    print("  - test_results.txt (human-readable results)")
    print("  - test_results.pkl (detailed results for analysis)")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history plot saved to './training_history.png'")

if __name__ == "__main__":
    main()