#!/usr/bin/env python3
"""
Baseline CNN Model for CAPTCHA Recognition
This script implements a CNN model to recognize text in CAPTCHA images using PyTorch.
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
from sklearn.metrics import precision_recall_fscore_support, classification_report
import string
import re
from tqdm import tqdm
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CaptchaDataProcessor:
    """Process CAPTCHA images and labels for training"""
    
    def __init__(self, img_width=200, img_height=80):
        self.img_width = img_width
        self.img_height = img_height
        self.characters = string.digits + string.ascii_lowercase  # 0-9, a-z
        # Add blank token for CTC at index 0
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.char_to_idx['<blank>'] = 0  # CTC blank token
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.characters) + 1  # +1 for blank token
        self.max_length = 12  # Maximum observed length with some buffer
        
    def extract_label_from_filename(self, filename):
        """Extract the CAPTCHA text from filename"""
        # Remove .png extension and -0 suffix
        basename = filename.replace('.png', '').replace('-0', '')
        # Clean up any remaining numbers at the end and keep only alphanumeric
        label = re.sub(r'[^a-z0-9]', '', basename.lower())
        return label
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
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
        
        # Add channel dimension
        processed = np.expand_dims(normalized, axis=-1)
        
        return processed
    
    def encode_label(self, label):
        """Encode label to sequence of integers for CTC"""
        # For CTC, we don't need padding - just encode the actual characters
        encoded = [self.char_to_idx.get(char, 0) for char in label if char in self.char_to_idx]
        return encoded
    
    def decode_prediction(self, prediction):
        """Decode CTC prediction back to text"""
        # CTC decoding: remove blanks and consecutive duplicates
        text = ""
        prev_char = None
        for idx in prediction:
            if idx == 0:  # blank token
                prev_char = None
                continue
            char = self.idx_to_char.get(idx, '')
            if char != prev_char:  # Remove consecutive duplicates
                text += char
                prev_char = char
        return text
    
    def ctc_decode_batch(self, predictions, lengths=None):
        """Batch CTC decoding"""
        decoded_texts = []
        for i, pred in enumerate(predictions):
            if lengths is not None:
                pred = pred[:lengths[i]]
            decoded_text = self.decode_prediction(pred)
            decoded_texts.append(decoded_text)
        return decoded_texts
    
    def load_data(self, data_dir, split_validation=True):
        """Load and preprocess all images and labels"""
        data_dir = Path(data_dir)
        
        images = []
        labels = []
        label_lengths = []
        
        print(f"Loading data from {data_dir}...")
        
        # Get all PNG files
        image_files = list(data_dir.glob("*.png"))
        
        # First pass: collect all labels to analyze lengths
        all_labels = []
        for img_path in image_files:
            label_text = self.extract_label_from_filename(img_path.name)
            if label_text:
                all_labels.append(label_text)
        
        # Print statistics about label lengths
        lengths = [len(label) for label in all_labels]
        print(f"Label length statistics:")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
        print(f"  Mean: {np.mean(lengths):.2f}, Median: {np.median(lengths):.2f}")
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # Preprocess image
            processed_img = self.preprocess_image(img_path)
            if processed_img is None:
                continue
                
            # Extract label from filename
            label_text = self.extract_label_from_filename(img_path.name)
            if not label_text:  # Skip if no valid label
                continue
                
            encoded_label = self.encode_label(label_text)
            if not encoded_label:  # Skip if encoding failed
                continue
                
            images.append(processed_img)
            labels.append(encoded_label)
            label_lengths.append(len(encoded_label))
        
        print(f"Loaded {len(images)} images")
        
        if split_validation:
            # Split into train and validation sets
            X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
                images, labels, label_lengths, test_size=0.2, random_state=42
            )
            return X_train, X_val, y_train, y_val, len_train, len_val
        else:
            return images, labels, label_lengths

class CaptchaDataset(Dataset):
    """PyTorch Dataset for CAPTCHA images with variable-length labels"""
    
    def __init__(self, images, labels, label_lengths, transform=None):
        self.images = images
        self.labels = labels
        self.label_lengths = label_lengths
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label_length = self.label_lengths[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and add channel dimension if needed
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
            else:
                image = np.transpose(image, (2, 0, 1))
            image = torch.FloatTensor(image)
        
        label = torch.LongTensor(label)
        return image, label, label_length

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    images, labels, lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad labels to max length in batch
    max_len = max(lengths)
    padded_labels = []
    for label in labels:
        padded = list(label) + [0] * (max_len - len(label))  # Pad with blank tokens
        padded_labels.append(padded)
    
    labels = torch.LongTensor(padded_labels)
    lengths = torch.LongTensor(lengths)
    
    return images, labels, lengths

class CaptchaCNN(nn.Module):
    """CNN + RNN Model for CAPTCHA Recognition using CTC Loss"""
    
    def __init__(self, img_width=200, img_height=80, num_classes=37):
        super(CaptchaCNN, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes  # 36 characters + 1 blank token
        
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
        # After 4 pooling operations: height/16, width/16
        self.feature_height = img_height // 16
        self.feature_width = img_width // 16
        
        # RNN for sequence modeling
        self.rnn_input_size = 256 * self.feature_height
        self.rnn_hidden_size = 256
        self.rnn = nn.LSTM(self.rnn_input_size, self.rnn_hidden_size, 
                          num_layers=2, batch_first=True, bidirectional=True)
        
        # Output layer for CTC
        self.output = nn.Linear(self.rnn_hidden_size * 2, num_classes)  # *2 for bidirectional
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Reshape for RNN: (batch, width, height*channels)
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.contiguous().view(batch_size, self.feature_width, -1)
        
        # RNN
        x, _ = self.rnn(x)
        
        # Output layer
        x = self.output(x)  # (batch, seq_len, num_classes)
        
        # For CTC, we need (seq_len, batch, num_classes)
        x = x.permute(1, 0, 2)
        
        return x

class CaptchaTrainer:
    """Training class for CAPTCHA CNN with CTC Loss"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for images, labels, label_lengths in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            log_probs = self.model(images)
            log_probs = torch.log_softmax(log_probs, dim=2)
            
            # Calculate input lengths (sequence length for each sample)
            input_lengths = torch.full(size=(images.size(0),), 
                                     fill_value=log_probs.size(0), 
                                     dtype=torch.long, device=self.device)
            
            # Flatten labels for CTC
            targets = []
            for i in range(labels.size(0)):
                targets.extend(labels[i][:label_lengths[i]].tolist())
            targets = torch.LongTensor(targets).to(self.device)
            
            # CTC Loss
            loss = self.criterion(log_probs, targets, input_lengths, label_lengths)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, labels, label_lengths in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                
                # Forward pass
                log_probs = self.model(images)
                log_probs = torch.log_softmax(log_probs, dim=2)
                
                # Calculate input lengths
                input_lengths = torch.full(size=(images.size(0),), 
                                         fill_value=log_probs.size(0), 
                                         dtype=torch.long, device=self.device)
                
                # Flatten labels for CTC
                targets = []
                for i in range(labels.size(0)):
                    targets.extend(labels[i][:label_lengths[i]].tolist())
                targets = torch.LongTensor(targets).to(self.device)
                
                # CTC Loss
                loss = self.criterion(log_probs, targets, input_lengths, label_lengths)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=30):
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_captcha_model.pth')
                print("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def predict(self, dataloader, processor, device='cpu'):
        """Make predictions using CTC decoding"""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for images, _, _ in dataloader:
                images = images.to(device)
                
                # Forward pass
                log_probs = self.model(images)
                log_probs = torch.log_softmax(log_probs, dim=2)
                
                # Simple greedy CTC decoding
                predictions = torch.argmax(log_probs, dim=2)  # (seq_len, batch)
                predictions = predictions.permute(1, 0)  # (batch, seq_len)
                
                for pred in predictions:
                    pred_text = processor.decode_prediction(pred.cpu().numpy())
                    all_predictions.append(pred_text)
        
        return all_predictions
    
    def evaluate_accuracy(self, dataloader, true_labels, processor, device='cpu'):
        """Evaluate model accuracy"""
        predictions = self.predict(dataloader, processor, device)
        
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for i in range(total_predictions):
            pred_text = predictions[i]
            true_text = processor.decode_prediction(true_labels[i])
            
            if pred_text == true_text:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        return accuracy, correct_predictions, total_predictions, predictions
    
    def evaluate_comprehensive(self, dataloader, true_labels, processor, device='cpu'):
        """Comprehensive evaluation with multiple metrics"""
        predictions = self.predict(dataloader, processor, device)
        
        # CAPTCHA-level metrics
        correct_captchas = 0
        total_captchas = len(predictions)
        
        # Character-level metrics
        true_chars = []
        pred_chars = []
        char_correct = 0
        char_total = 0
        
        # For precision/recall calculation
        all_true_texts = []
        all_pred_texts = []
        
        for i in range(total_captchas):
            pred_text = predictions[i]
            true_text = processor.decode_prediction(true_labels[i])
            
            all_true_texts.append(true_text)
            all_pred_texts.append(pred_text)
            
            # CAPTCHA-level accuracy
            if pred_text == true_text:
                correct_captchas += 1
            
            # Character-level accuracy
            max_len = max(len(true_text), len(pred_text))
            for j in range(max_len):
                true_char = true_text[j] if j < len(true_text) else '<pad>'
                pred_char = pred_text[j] if j < len(pred_text) else '<pad>'
                
                if j < len(true_text):  # Only count actual characters, not padding
                    char_total += 1
                    true_chars.append(true_char)
                    pred_chars.append(pred_char)
                    
                    if true_char == pred_char:
                        char_correct += 1
        
        # Calculate metrics
        captcha_accuracy = correct_captchas / total_captchas
        char_accuracy = char_correct / char_total if char_total > 0 else 0
        
        # Calculate precision, recall, F1 for each unique character
        unique_chars = list(set(true_chars + pred_chars))
        
        # Convert characters to indices for sklearn
        char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        true_char_indices = [char_to_idx[char] for char in true_chars]
        pred_char_indices = [char_to_idx[char] for char in pred_chars]
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_char_indices, pred_char_indices, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_char_indices, pred_char_indices, average='macro', zero_division=0
        )
        
        results = {
            'captcha_accuracy': captcha_accuracy,
            'captcha_correct': correct_captchas,
            'captcha_total': total_captchas,
            'char_accuracy': char_accuracy,
            'char_correct': char_correct,
            'char_total': char_total,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'predictions': predictions,
            'true_texts': all_true_texts,
            'pred_texts': all_pred_texts
        }
        
        return results

def visualize_predictions(X_test, y_test, predictions, processor, num_samples=10):
    """Visualize predictions"""
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(min(num_samples, len(X_test))):
        plt.subplot(num_samples, 1, i + 1)
        
        # Display image
        img = X_test[i].squeeze()
        plt.imshow(img, cmap='gray')
        
        # Get texts
        pred_text = processor.decode_prediction(predictions[i])
        true_text = processor.decode_prediction(y_test[i])
        
        # Set title
        color = 'green' if pred_text == true_text else 'red'
        plt.title(f'True: {true_text}, Predicted: {pred_text}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sample predictions saved to './sample_predictions.png'")

def main():
    """Main function"""
    # Configuration
    TRAIN_DIR = "./train"
    TEST_DIR = "./test"
    BATCH_SIZE = 32
    EPOCHS = 200
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=== CAPTCHA CNN Baseline Model (PyTorch) ===")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    
    # Initialize data processor
    processor = CaptchaDataProcessor()
    
    # Load training data
    print("\nLoading training data...")
    X_train, X_val, y_train, y_val, len_train, len_val = processor.load_data(TRAIN_DIR, split_validation=True)
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test, len_test = processor.load_data(TEST_DIR, split_validation=False)
    print(f"Test set: {len(X_test)} images")
    
    # Create datasets and dataloaders
    train_dataset = CaptchaDataset(X_train, y_train, len_train)
    val_dataset = CaptchaDataset(X_val, y_val, len_val)
    test_dataset = CaptchaDataset(X_test, y_test, len_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Build model
    print("\nBuilding model...")
    model = CaptchaCNN(num_classes=processor.num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer
    trainer = CaptchaTrainer(model, device)
    
    print("\nTraining model...")
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Load best model (selected based on validation loss)
    print("\nLoading best model (selected based on validation loss)...")
    model.load_state_dict(torch.load('best_captcha_model.pth'))
    trainer.model = model.to(device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate_comprehensive(test_loader, y_test, processor, device)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    
    print(f"\nCAPTCHA-Level Metrics:")
    print(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})")
    
    print(f"\nCharacter-Level Metrics:")
    print(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {results['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro): {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro): {results['f1_macro']:.4f}")
    
    # Show some example predictions
    print(f"\nSample Predictions:")
    predictions = results['predictions']
    for i in range(min(10, len(predictions))):
        true_text = processor.decode_prediction(y_test[i])
        pred_text = predictions[i]
        status = "✓" if pred_text == true_text else "✗"
        print(f"  {status} True: '{true_text}' | Pred: '{pred_text}'")
    
    # Convert predictions back to arrays for visualization
    pred_arrays = []
    for pred_text in predictions:
        pred_array = processor.encode_label(pred_text)
        # Pad to same length as original for visualization
        while len(pred_array) < processor.max_length:
            pred_array.append(0)
        pred_arrays.append(pred_array[:processor.max_length])
    
    # Visualize some predictions
    print("\nVisualizing predictions...")
    visualize_predictions(X_test, y_test, pred_arrays, processor, num_samples=10)
    
    # Save detailed results
    print("\nSaving results...")
    
    # Save metrics to text file
    with open('./test_results.txt', 'w') as f:
        f.write("CAPTCHA Recognition Test Results\n")
        f.write("="*40 + "\n\n")
        
        f.write("CAPTCHA-Level Metrics:\n")
        f.write(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})\n\n")
        
        f.write("Character-Level Metrics:\n")
        f.write(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})\n")
        f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {results['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
        f.write(f"  Recall (macro): {results['recall_macro']:.4f}\n")
        f.write(f"  F1-Score (macro): {results['f1_macro']:.4f}\n\n")
        
        f.write("Sample Predictions:\n")
        for i in range(min(20, len(predictions))):
            true_text = processor.decode_prediction(y_test[i])
            pred_text = predictions[i]
            status = "✓" if pred_text == true_text else "✗"
            f.write(f"  {status} True: '{true_text}' | Pred: '{pred_text}'\n")
    
    # Save model and processor
    torch.save(model.state_dict(), 'captcha_cnn_model.pth')
    
    with open('captcha_processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    # Save results as pickle for further analysis
    with open('test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Model, processor, and results saved successfully!")
    print("Files saved:")
    print("  - captcha_cnn_model.pth (model weights)")
    print("  - captcha_processor.pkl (data processor)")
    print("  - test_results.txt (human-readable results)")
    print("  - test_results.pkl (detailed results for analysis)")
    
    # Plot training history
    plt.figure(figsize=(8, 6))
    
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('CTC Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history plot saved to './training_history.png'")

if __name__ == "__main__":
    main()
