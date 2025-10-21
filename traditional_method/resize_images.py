import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def process_single_image(args):
    """
    Process a single image for resizing.
    
    Args:
        args (tuple): (image_file_path, output_file_path, target_width, target_height)
    
    Returns:
        tuple: (success, image_file_path, error_message)
    """
    image_file_path, output_file_path, target_width, target_height = args
    
    try:
        # Read the image
        image = cv2.imread(str(image_file_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return (False, image_file_path, "Could not read image")
        
        # Resize the image
        resized_image = cv2.resize(image, (target_width, target_height), 
                                 interpolation=cv2.INTER_AREA)
        
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the resized image
        cv2.imwrite(str(output_file_path), resized_image)
        
        return (True, image_file_path, None)
        
    except Exception as e:
        return (False, image_file_path, str(e))


def resize_images_in_folder(input_folder, output_folder, target_width=28, target_height=28, num_processes=None):
    """
    Resize all images in subfolders to specified dimensions using multiprocessing.
    
    Args:
        input_folder (str): Path to the input folder containing subfolders with images
        output_folder (str): Path to the output folder where resized images will be saved
        target_width (int): Target width for resized images
        target_height (int): Target height for resized images
        num_processes (int): Number of processes to use (default: CPU count)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Collect all image processing tasks
    image_tasks = []
    total_images = 0
    
    # Walk through all subdirectories and collect image files
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            # Find all image files in the subfolder
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            image_files = [f for f in subfolder.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            total_images += len(image_files)
            
            print(f"Found subfolder: {subfolder.name} ({len(image_files)} images)")
            
            # Create tasks for each image
            for image_file in image_files:
                output_subfolder = output_path / subfolder.name
                output_filename = f"{image_file.stem}_resized.png"
                output_filepath = output_subfolder / output_filename
                
                task = (image_file, output_filepath, target_width, target_height)
                image_tasks.append(task)
    
    if not image_tasks:
        print("No images found to process.")
        return
    
    print(f"\nStarting parallel processing of {total_images} images...")
    start_time = time.time()
    
    # Process images in parallel
    successful_processes = 0
    failed_processes = 0
    
    with Pool(processes=num_processes) as pool:
        # Process images and show progress
        results = []
        
        # Submit all tasks
        for i in range(0, len(image_tasks), num_processes * 10):  # Process in batches
            batch = image_tasks[i:i + num_processes * 10]
            batch_results = pool.map(process_single_image, batch)
            results.extend(batch_results)
            
            # Update progress
            processed_so_far = min(i + len(batch), len(image_tasks))
            print(f"Progress: {processed_so_far}/{len(image_tasks)} images processed", end='\r')
    
    # Count results
    for success, image_path, error in results:
        if success:
            successful_processes += 1
        else:
            failed_processes += 1
            print(f"\nError processing {image_path}: {error}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n\nCompleted! Processed {successful_processes}/{total_images} images successfully.")
    if failed_processes > 0:
        print(f"Failed to process {failed_processes} images.")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/total_images:.4f} seconds")
    print(f"Resized images saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images in subfolders to specified dimensions using parallel processing')
    parser.add_argument('input_folder', type=str,
                       help='Input folder containing subfolders with images')
    parser.add_argument('--width', type=int, default=28,
                       help='Target width for resized images (default: 28)')
    parser.add_argument('--height', type=int, default=28,
                       help='Target height for resized images (default: 28)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output folder name (default: <input_folder>_resized_<width>x<height>)')
    parser.add_argument('--processes', type=int, default=None,
                       help=f'Number of processes to use (default: {cpu_count()}, available: {cpu_count()})')
    
    args = parser.parse_args()
    
    # Determine output folder name
    if args.output is None:
        input_folder_name = Path(args.input_folder).name
        output_folder = f"{args.input_folder}_resized_{args.width}x{args.height}"
    else:
        output_folder = args.output
    
    # Validate processes argument
    max_processes = cpu_count()
    if args.processes is not None:
        if args.processes < 1:
            print(f"Error: Number of processes must be at least 1")
            exit(1)
        elif args.processes > max_processes:
            print(f"Warning: Requested {args.processes} processes, but only {max_processes} CPU cores available")
            print(f"Using {max_processes} processes instead")
            args.processes = max_processes
    
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Target dimensions: {args.width}x{args.height}")
    print(f"Available CPU cores: {max_processes}")
    print("-" * 50)
    
    resize_images_in_folder(args.input_folder, output_folder, args.width, args.height, args.processes)
