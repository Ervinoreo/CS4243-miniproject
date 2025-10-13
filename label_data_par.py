import os
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm


def process_single_folder(item, output_path, lock):
    """
    Process a single folder to rename and organize files.
    
    Args:
        item (Path): Path to the folder to process
        output_path (Path): Path to the output folder where files will be organized
        lock (threading.Lock): Lock for thread-safe printing and file operations
    
    Returns:
        dict: Statistics about the processing of this folder
    """
    folder_name = item.name
    
    # Extract name part from format <name>-0
    if folder_name.endswith("-0"):
        name_part = folder_name[:-2]  # Remove "-0" suffix
    else:
        # If folder doesn't follow the expected format, skip processing
        return {"processed": False, "folder_name": folder_name}
    
    # Get all files in the folder
    files = [f for f in item.iterdir() if f.is_file()]
    file_count = len(files)
    name_length = len(name_part)
    
    result = {
        'processed': False,
        'files_processed': 0,
        'folder_name': folder_name
    }
    
    # Check if name length equals file count
    if name_length == file_count:
        
        result['processed'] = True
        
        # Process each file
        for i, file_path in enumerate(files):
            try:
                # Get the filename without extension
                file_stem = file_path.stem
                
                # Extract idx from filename (first 3 characters, e.g., "000" from "000_char.png")
                idx = file_stem[:3] if len(file_stem) >= 3 else file_stem
                
                # Convert idx to integer to get the position in folder name
                try:
                    idx_num = int(idx)
                    if idx_num < len(name_part):
                        # Get the character at position idx_num from the folder name
                        char_at_idx = name_part[idx_num]
                        
                        # Create output subfolder named after the character
                        output_subfolder = output_path / char_at_idx
                        
                        # Thread-safe folder creation
                        with lock:
                            output_subfolder.mkdir(parents=True, exist_ok=True)
                        
                        # Create new filename
                        new_filename = f"{name_part}_{idx}.png"
                        output_file_path = output_subfolder / new_filename
                        
                        # Copy the file to the new location with new name
                        shutil.copy2(file_path, output_file_path)
                        
                        result['files_processed'] += 1
                    else:
                        with lock:
                            print(f"    Skipping {file_path.name}: idx {idx_num} exceeds folder name length {len(name_part)}")
                except ValueError:
                    with lock:
                        print(f"    Skipping {file_path.name}: invalid idx '{idx}' (not a number)")

            except Exception as e:
                with lock:
                    print(f"    Error processing file {file_path.name}: {e}")
    
    return result


def process_folders(input_folder, output_folder, num_threads=4):
    """
    Process folders to rename and organize files based on folder name length using parallel processing.
    
    Args:
        input_folder (str): Path to the input folder containing folders to process
        output_folder (str): Path to the output folder where files will be organized
        num_threads (int): Number of threads to use for parallel processing
    
    Returns:
        dict: Statistics about the processing
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_path}' does not exist.")
        return None
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all folders to process (excluding debug folder)
    folders_to_process = [
        item for item in input_path.iterdir() 
        if item.is_dir() and item.name != "debug"
    ]
    
    total_folders = len(folders_to_process)
    processed_folders = 0
    skipped_folders = 0
    total_files_processed = 0
    skipped_folder_names = []  # List to track skipped folder names
    
    print(f"Found {total_folders} folders to process using {num_threads} threads...")
    
    # Create a lock for thread-safe operations
    lock = threading.Lock()
    
    # Process folders in parallel with progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_folder = {
            executor.submit(process_single_folder, folder, output_path, lock): folder 
            for folder in folders_to_process
        }
        
        # Create progress bar
        with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
            # Collect results as they complete
            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    result = future.result()
                    if result['processed']:
                        processed_folders += 1
                        total_files_processed += result['files_processed']
                    else:
                        skipped_folders += 1
                        skipped_folder_names.append(result['folder_name'])
                except Exception as exc:
                    with lock:
                        tqdm.write(f"Folder {folder.name} generated an exception: {exc}")
                finally:
                    pbar.update(1)
    
    # Write skipped folder names to a text file
    if skipped_folder_names:
        skipped_file_path = output_path / "skipped_folders.txt"
        with open(skipped_file_path, 'w') as f:
            f.write("Skipped folders (length mismatch):\n")
            f.write("=" * 40 + "\n")
            for folder_name in sorted(skipped_folder_names):
                f.write(f"{folder_name}\n")
        print(f"Skipped folders list saved to: {skipped_file_path}")
    
    # Print summary statistics
    print("=" * 60)
    print("PARALLEL PROCESSING SUMMARY:")
    print(f"Threads used: {num_threads}")
    print(f"Total folders found: {total_folders}")
    print(f"Folders processed: {processed_folders}")
    print(f"Folders skipped (length mismatch): {skipped_folders}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Output folder: {output_path}")
    if skipped_folder_names:
        print(f"Skipped folders list: {output_path / 'skipped_folders.txt'}")
    print("=" * 60)
    
    return {
        'total_folders': total_folders,
        'processed_folders': processed_folders,
        'skipped_folders': skipped_folders,
        'total_files_processed': total_files_processed
    }


def main():
    """Main function to handle command line arguments and run the processing."""
    parser = argparse.ArgumentParser(
        description="Process folders to rename and organize files based on folder name length using parallel processing"
    )
    parser.add_argument(
        "input_folder",
        help="Path to the input folder containing folders to process"
    )
    parser.add_argument(
        "output_folder",
        help="Path to the output folder where files will be organized"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Validate number of threads
    if args.threads < 1:
        print("Error: Number of threads must be at least 1.")
        return 1
    
    # Run the processing
    result = process_folders(args.input_folder, args.output_folder, args.threads)
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
