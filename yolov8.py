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
    """
    
    # Initialize model
    model_name = f"yolov8{model_size}.pt"
    print(f"\n=== Initializing YOLOv8{model_size.upper()} ===")
    model = YOLO(model_name)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Training configuration (removed split since you have separate val folder)
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
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        # Augmentation settings
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        # Training settings
        'val': True,
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': 8,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'patience': 50,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
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
    MODEL_SIZE = "l"  # Start with nano for faster training
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
        
        print(f"\n=== Training Pipeline Completed ===")
        print(f"Best model: {best_model_path}")
        print(f"Results directory: {model.trainer.save_dir}")
        print(f"\nTo make predictions on new images:")
        print(f"yolo predict model={best_model_path} source=path/to/images")
        print(f"\nTo run validation:")
        print(f"yolo val model={best_model_path} data={DATA_YAML}")
        
    else:
        print("Training failed. Please check the error messages above.")

        
if __name__ == "__main__":
    main()