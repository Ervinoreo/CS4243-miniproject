#!/usr/bin/env python3
"""
Character-Level VGG16 Model for CAPTCHA Recognition
This script implements a VGG16 model to classify individual characters, then aggregates them for CAPTCHA-level evaluation.
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
import torchvision.models as models
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
    
    def __init__(self, img_width=224, img_height=224):
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
        """Preprocess a single character image for VGG16"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Resize image to 224x224 (VGG16 input size)
        resized = cv2.resize(rgb, (self.img_width, self.img_height))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to CHW format (channels first)
        normalized = np.transpose(normalized, (2, 0, 1))
        
        return normalized
    
    def load_character_data(self, data_dir, split_validation=True):
        """Load character images organized by class folders"""
        data_dir = Path(data_dir)
        
        images = []
        labels = []
        image_paths = []  # Store paths for filename-based evaluation
        
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
                
                images.append(processed_img)
                labels.append(class_idx)
                image_paths.append(str(img_path))  # Store full path for evaluation
        
        print(f"Loaded {len(images)} character images")
        print(f"Character distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            char = self.idx_to_char[label_idx]
            print(f"  '{char}': {count} samples")
        
        if split_validation:
            # Split into train and validation sets
            X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
                images, labels, image_paths, test_size=0.2, random_state=42, stratify=labels
            )
            return X_train, X_val, y_train, y_val, paths_train, paths_val
        else:
            return images, labels, image_paths

class CharacterDataset(Dataset):
    """PyTorch Dataset for individual character images"""
    
    def __init__(self, images, labels, image_paths=None, transform=None):
        self.images = images
        self.labels = labels
        self.image_paths = image_paths  # Store paths for filename-based evaluation
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Convert CHW to HWC for transforms, then transforms will convert back to tensor
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            image = (image * 255).astype(np.uint8)  # Convert back to 0-255 range for PIL
            image = self.transform(image)
        else:
            # Convert to tensor (already in CHW format from preprocessing)
            image = torch.FloatTensor(image)
            # Apply ImageNet normalization manually
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
        
        label = torch.LongTensor([label])[0]  # Convert to scalar tensor
        
        return image, label

def simple_collate_fn(batch):
    """Simple collate function without metadata complexity"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

class CharacterVGG16(nn.Module):
    """VGG16-based Model for Individual Character Classification"""
    
    def __init__(self, num_classes=36, pretrained=True):
        super(CharacterVGG16, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Get the number of features from VGG16's classifier
        num_features = self.vgg.classifier[0].in_features
        
        # Replace the classifier (last 3 layers)
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
        
        # Optionally freeze early layers for fine-tuning
        self.freeze_early_layers = True
        if self.freeze_early_layers:
            # Freeze all convolutional layers (features)
            for param in self.vgg.features.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        return self.vgg(x)
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for full fine-tuning"""
        for param in self.vgg.parameters():
            param.requires_grad = True

class CharacterTrainer:
    """Training class for Character VGG16"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Use different learning rates for pretrained and new layers
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name:  # New classifier layers
                new_params.append(param)
            else:  # Pretrained features (conv layers)
                pretrained_params.append(param)
        
        # Different learning rates for pretrained vs new layers
        self.optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': 0.00001},  # Lower lr for pretrained
            {'params': new_params, 'lr': 0.001}  # Higher lr for new layers
        ], weight_decay=1e-4)
        
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
                torch.save(self.model.state_dict(), 'best_character_vgg16_model.pth')
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
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_probabilities
    
    def aggregate_to_captcha_predictions(self, char_predictions, image_paths, processor):
        """Aggregate character predictions to CAPTCHA-level predictions using filenames"""
        captcha_predictions = defaultdict(list)
        
        # Group predictions by CAPTCHA ID using filename parsing
        for pred, img_path in zip(char_predictions, image_paths):
            # Extract filename
            filename = Path(img_path).name
            
            # Parse filename to get CAPTCHA ID and position
            captcha_id, position = processor.parse_filename(filename)
            if captcha_id is None:
                continue
            
            # Get predicted character
            predicted_char = processor.idx_to_char[pred]
            
            # Get true character from folder name (parent directory)
            true_char = Path(img_path).parent.name
            
            captcha_predictions[captcha_id].append((position, predicted_char, true_char))
        
        # Sort by position and create final CAPTCHA predictions
        final_predictions = {}
        true_captchas = {}
        
        for captcha_id, char_data in captcha_predictions.items():
            # Sort by position
            char_data.sort(key=lambda x: x[0])
            
            # Extract predicted and true sequences
            pred_sequence = ''.join([char_info[1] for char_info in char_data])
            true_sequence = ''.join([char_info[2] for char_info in char_data])
            
            final_predictions[captcha_id] = pred_sequence
            true_captchas[captcha_id] = true_sequence
        
        return final_predictions, true_captchas
    
    def evaluate_comprehensive(self, test_dataset, processor, device='cpu'):
        """Comprehensive evaluation with both character and CAPTCHA level metrics"""
        # Create test dataloader
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=simple_collate_fn)
        
        # Get character predictions
        char_predictions, char_probabilities = self.predict_characters(test_loader, device)
        
        # Get true labels from dataset
        true_char_labels = test_dataset.labels
        
        # Character-level evaluation
        char_accuracy = accuracy_score(true_char_labels, char_predictions)
        
        # Calculate precision, recall, F1 for character level
        precision, recall, f1, support = precision_recall_fscore_support(
            true_char_labels, char_predictions, average='weighted', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_char_labels, char_predictions, average='macro', zero_division=0
        )
        
        # CAPTCHA-level evaluation using image paths
        captcha_predictions, true_captchas = self.aggregate_to_captcha_predictions(
            char_predictions, test_dataset.image_paths, processor
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

            'captcha_predictions': captcha_predictions,
            'true_captchas': true_captchas
        }
        
        return results

def visualize_character_predictions(X_test, y_test, predictions, processor, num_samples=20):
    """Visualize character predictions"""
    plt.figure(figsize=(20, 4 * (num_samples // 5)))
    
    for i in range(min(num_samples, len(X_test))):
        plt.subplot(num_samples // 5, 5, i + 1)
        
        # Display image (convert from CHW to HWC for visualization)
        img = X_test[i]
        if len(img.shape) == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
        plt.imshow(img)
        
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
    BATCH_SIZE = 32  # Batch size for VGG16
    EPOCHS = 50  # Epochs for pretrained model
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=== Character-Level VGG16 Model (PyTorch) ===")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    
    # Initialize data processor
    processor = CharacterDataProcessor()
    
    # Load training data
    print("\nLoading training data...")
    X_train, X_val, y_train, y_val, paths_train, paths_val = processor.load_character_data(TRAIN_DIR, split_validation=True)
    
    print(f"Training set: {len(X_train)} character images")
    print(f"Validation set: {len(X_val)} character images")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test, paths_test = processor.load_character_data(TEST_DIR, split_validation=False)
    print(f"Test set: {len(X_test)} character images")
    
    # Define transforms for data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Create datasets and dataloaders
    train_dataset = CharacterDataset(X_train, y_train, paths_train, transform=train_transform)
    val_dataset = CharacterDataset(X_val, y_val, paths_val, transform=val_transform)
    test_dataset = CharacterDataset(X_test, y_test, paths_test, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=simple_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=simple_collate_fn)
    
    # Build model
    print("\nBuilding model...")
    model = CharacterVGG16(num_classes=processor.num_classes, pretrained=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Initialize trainer
    trainer = CharacterTrainer(model, device)
    
    print("\nTraining model...")
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_character_vgg16_model.pth'))
    trainer.model = model.to(device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate_comprehensive(test_dataset, processor, device)
    
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
    with open('./test_results_vgg16.txt', 'w') as f:
        f.write("Character-Level VGG16 Test Results\n")
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
    with open('./test_results_vgg16.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Files saved:")
    print("  - test_results_vgg16.txt (human-readable results)")
    print("  - test_results_vgg16.pkl (detailed results for analysis)")
    
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