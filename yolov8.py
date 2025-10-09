#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline for CAPTCHA Character Detection
This script trains a YOLOv8 model to detect individual characters in CAPTCHA images.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def train_yolov8_captcha(
    data_yaml_path,
    model_size="n",  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    project="captcha_yolo",
    name="yolov8_captcha",
    resume=False,
    device=None
):
    """
    Train YOLOv8 model for CAPTCHA character detection
    
    Args:
        data_yaml_path: Path to data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        project: Project directory name
        name: Experiment name
        resume: Resume from last checkpoint
        device: Device to use (cuda/cpu/auto)
    """
    
    
    # Initialize model
    model_name = f"yolov8{model_size}.pt"
    print(f"\n=== Initializing YOLOv8{model_size.upper()} ===")
    model = YOLO(model_name)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Training configuration
    train_config = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'project': project,
        'name': name,
        'device': device,
        'resume': resume,
        # Optimization settings
        'optimizer': 'AdamW',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.1,   # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        # Augmentation settings
        'hsv_h': 0.015,  # HSV hue augmentation
        'hsv_s': 0.7,    # HSV saturation augmentation
        'hsv_v': 0.4,    # HSV value augmentation
        'degrees': 0.0,  # Rotation (degrees)
        'translate': 0.1, # Translation
        'scale': 0.5,    # Scale
        'shear': 0.0,    # Shear
        'perspective': 0.0, # Perspective
        'flipud': 0.0,   # Vertical flip
        'fliplr': 0.5,   # Horizontal flip
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.0,    # Mixup augmentation
        'copy_paste': 0.0, # Copy-paste augmentation
        # Validation settings
        'val': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every N epochs
        'cache': False,  # Cache images for faster training
        'workers': 8,    # Number of worker threads
        'verbose': True,
        'seed': 42,      # Random seed for reproducibility
        'deterministic': True,
        # Early stopping
        'patience': 50,  # Epochs to wait for no improvement
        # Other settings
        'single_cls': False,  # Treat as single-class dataset
        'rect': False,   # Rectangular training
        'cos_lr': False, # Cosine learning rate scheduler
        'close_mosaic': 10, # Disable mosaic in final epochs
    }
    
    print(f"\n=== Training Configuration ===")
    for key, value in train_config.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Starting Training ===")
    
    try:
        # Train the model
        results = model.train(**train_config)
        
        print(f"\n=== Training Completed Successfully ===")
        print(f"Best weights saved to: {model.trainer.best}")
        print(f"Last weights saved to: {model.trainer.last}")
        print(f"Results saved to: {model.trainer.save_dir}")
        
        return results, model
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None, None

def perform_inference(model_path, train_images_dir, output_dir="inference", num_images=5):
    """Perform inference on first N training images and save results with bounding boxes"""
    print(f"\n=== Performing Inference on Training Images ===")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load trained model
    model = YOLO(model_path)
    
    # Get first N training images
    train_images_path = Path(train_images_dir)
    image_files = list(train_images_path.glob("*.jpg"))[:num_images]
    
    if len(image_files) == 0:
        print(f"No images found in {train_images_dir}")
        return
    
    print(f"Running inference on {len(image_files)} images...")
    print(f"Saving results to: {output_path.absolute()}")
    
    # Run inference and save results
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
        
        # Run inference
        results = model(str(image_path), conf=0.25, iou=0.6)
        
        # Save image with bounding boxes
        output_file = output_path / f"inference_{i+1}_{image_path.name}"
        
        # Plot and save the results
        for result in results:
            result.save(filename=str(output_file))
        
        print(f"Saved: {output_file}")
    
    print(f"\n=== Inference completed! Check {output_path} folder ===")
    return output_path


def export_model(model_path, format='onnx'):
    """Export model to different formats"""
    print(f"\n=== Exporting Model to {format.upper()} ===")
    model = YOLO(model_path)
    
    exported_model = model.export(format=format)
    print(f"Model exported to: {exported_model}")
    
    return exported_model

def main():
    """Main training pipeline"""
    # Configuration
    DATA_YAML = "/home/e/ervin/cv4243-working/CAPTCHA.v1-v1.yolov8/data.yaml"
    TRAIN_IMAGES_DIR = "/home/e/ervin/cv4243-working/data/train/"
    MODEL_SIZE = "n"  # Start with nano for faster training
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    PROJECT = "captcha_detection"
    EXPERIMENT_NAME = "yolov8n_captcha_v1"
    
    print("=== YOLOv8 CAPTCHA Character Detection Training ===")
    print(f"Data: {DATA_YAML}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}")
    
    # Check if data.yaml exists
    if not os.path.exists(DATA_YAML):
        print(f"Error: data.yaml not found at {DATA_YAML}")
        print("Please check the path and try again.")
        return
    
    # Train model
    results, model = train_yolov8_captcha(
        data_yaml_path=DATA_YAML,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        resume=False
    )
    
    if results is not None:
        # Get best model path
        best_model_path = model.trainer.best
        
        # Perform inference on first 5 training images
        inference_output = perform_inference(
            model_path=best_model_path,
            train_images_dir=TRAIN_IMAGES_DIR,
            output_dir="./inference",
            num_images=5
        )
        
        # Export model (optional)
        try:
            export_model(best_model_path, format='onnx')
        except Exception as e:
            print(f"Export failed: {e}")
        
        print(f"\n=== Training Pipeline Completed ===")
        print(f"Best model: {best_model_path}")
        print(f"Inference results: {inference_output}")
        print(f"Use this command to make predictions:")
        print(f"yolo predict model={best_model_path} source=path/to/test/images")
        
    else:
        print("Training failed. Please check the error messages above.")
if __name__ == "__main__":
    main()